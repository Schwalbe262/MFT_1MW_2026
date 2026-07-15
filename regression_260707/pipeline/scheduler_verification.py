"""Reviewed scheduler-client execution for pipeline verification manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


ADAPTER_NAME = "mft_scheduler_v1"
TERMINAL = {"completed", "failed", "cancelled"}


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=1, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _load_state(path: Path, request_sha256: str, candidates: list[dict]):
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("request_sha256") != request_sha256:
            raise RuntimeError("scheduler verification request changed across retry")
        return state
    state = {
        "schema_version": 1,
        "request_sha256": request_sha256,
        "records": {
            item["candidate_id"]: {
                "rank": rank,
                "attempt": 0,
                "active_id": None,
                "outcome": "new",
            }
            for rank, item in enumerate(candidates)
        },
    }
    _atomic_json(state, path)
    return state


def _candidate_params(candidate):
    from module.input_parameter_260706 import KEYS

    raw = candidate.get("parameters")
    if not isinstance(raw, dict):
        raise RuntimeError("verification candidate parameters are unavailable")
    params = {key: raw[key] for key in KEYS if key in raw and raw[key] is not None}
    if set(params) != set(KEYS):
        raise RuntimeError(
            "verification candidate does not contain the sealed input schema"
        )
    return params


def run_scheduler_verification(request_path, result_path, config):
    """Submit/reconcile one exact manifest through the hardened scheduler client."""
    if not isinstance(config, dict) or set(config) - {
        "adapter", "execute", "library_root", "poll_seconds", "timeout_seconds"
    }:
        raise RuntimeError("unknown scheduler verification configuration field")
    if config.get("adapter") != ADAPTER_NAME or config.get("execute") is not True:
        raise RuntimeError("reviewed scheduler verification is not explicitly enabled")
    library_root = os.path.abspath(os.fspath(config.get("library_root") or ""))
    if not library_root or not os.path.isdir(library_root):
        raise RuntimeError("reviewed scheduler verification needs a library_root")

    request_path = Path(request_path).resolve()
    result_path = Path(result_path).resolve()
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    stage = request.get("stage")
    candidates = request.get("candidates")
    solver_revision = str(request.get("solver_revision") or "").lower()
    library_revision = str(request.get("library_revision") or "").lower()
    if stage not in ("standard", "fine") or not isinstance(candidates, list):
        raise RuntimeError("verification request schema is invalid")
    for label, revision in (
        ("solver", solver_revision), ("library", library_revision)
    ):
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise RuntimeError(f"verification request has no pinned {label} revision")

    repo_root = Path(__file__).resolve().parents[2]
    from campaign.deployment_gate import validate_deployment

    validate_deployment(
        repo_root, solver_revision, library_root, library_revision
    )
    from verify import scheduler_client as scheduler

    profile_path = repo_root / "regression_260707" / "verify" / "profiles" / f"{stage}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    state_path = result_path.parent / "scheduler_state.json"
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    state = _load_state(state_path, request_sha256, candidates)
    records = state["records"]
    by_id = {item["candidate_id"]: item for item in candidates}
    if set(records) != set(by_id):
        raise RuntimeError("scheduler task inventory does not match selection")

    def submit_record(candidate_id, record, *, retry):
        params = _candidate_params(by_id[candidate_id])
        submitted = scheduler.effective_verification_params(params, profile)
        rank = int(record["rank"])
        short = request["input_generation_id"][:10]
        base_name = f"mft-pipe-{stage}-{short}-{rank:02d}"
        base_workdir = f"mft_pipe_{stage}_{short}_{rank:02d}"
        name = f"{base_name}-retry" if retry else base_name
        workdir = f"{base_workdir}_retry" if retry else base_workdir
        submitting = "retry_submitting" if retry else "submitting"
        expected_attempt = 1 if retry else 0
        if record.get("outcome") != submitting:
            record.update(
                active_id=None,
                attempt=expected_attempt,
                outcome=submitting,
                submitted_params=submitted,
                name=name,
                workdir=workdir,
            )
            # Seal the deterministic scheduler identity before the call.  If
            # the process dies after POST, a queue retry invokes the hardened
            # scheduler client's exact name/dedupe reconciliation instead of
            # inventing a different task identity.
            _atomic_json(state, state_path)
        elif (
            record.get("name") != name
            or record.get("workdir") != workdir
            or int(record.get("attempt", -1)) != expected_attempt
            or record.get("submitted_params") != submitted
        ):
            raise RuntimeError("sealed scheduler submission identity changed")
        task_id = scheduler.submit_verification(
            name,
            workdir,
            params,
            profile,
            mem_mb=(
                max(
                    int(profile.get("mem_mb", 32768)),
                    131072 if stage == "fine" else 65536,
                )
                if retry else int(profile.get("mem_mb", 32768))
            ),
            cpus=int(profile.get("cpus", 4)),
            solver_revision=solver_revision,
            library_revision=library_revision,
        )
        if task_id is None:
            record["last_submission_error"] = "task_identity_unknown"
            _atomic_json(state, state_path)
            raise RuntimeError(
                f"scheduler submission identity is unknown: {candidate_id}; "
                "the sealed identity will be reconciled on retry"
            )
        record.update(active_id=task_id, outcome="pending")
        record.pop("last_submission_error", None)
        _atomic_json(state, state_path)
        time.sleep(0.05)

    for candidate_id, record in records.items():
        outcome = record.get("outcome")
        if outcome in ("new", "submitting"):
            submit_record(candidate_id, record, retry=False)
        elif outcome == "retry_submitting":
            submit_record(candidate_id, record, retry=True)

    poll_seconds = max(5, int(config.get("poll_seconds", 180)))
    timeout_seconds = max(60, int(config.get("timeout_seconds", 6 * 3600)))
    while True:
        pending = {
            candidate_id: record
            for candidate_id, record in records.items()
            if record.get("outcome") == "pending"
        }
        if not pending:
            break
        task_ids = [int(record["active_id"]) for record in pending.values()]
        statuses = scheduler.wait_all(
            task_ids, poll_s=poll_seconds, timeout_s=timeout_seconds
        )
        retried = False
        unresolved = False
        for candidate_id, record in pending.items():
            task_id = int(record["active_id"])
            status = statuses.get(task_id)
            record["last_status"] = status
            if status not in TERMINAL:
                unresolved = True
                continue
            fetched = scheduler.fetch_result(
                task_id,
                expected_revision=solver_revision,
                expected_library_revision=library_revision,
                expected_profile=profile.get("param_overrides"),
            )
            matches = (
                fetched.state == scheduler.RESULT_VALID
                and scheduler.result_matches_params(
                    fetched.result,
                    record["submitted_params"],
                    required_keys=record["submitted_params"].keys(),
                )
            )
            if matches:
                from optimization.geometry_metrics import bounding_box_lit
                from verify.finalize import physical_spec_reasons

                result = fetched.result
                record["result"] = result
                record["actual_volume_L"] = float(bounding_box_lit(result)[0])
                losses = (
                    result.get("P_winding_total"),
                    result.get("P_core_total"),
                    result.get("P_core_plate_total"),
                    result.get("P_wcp_total"),
                )
                if all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in losses
                ):
                    record["actual_total_loss_W"] = sum(float(value) for value in losses)
                record["valid"] = not physical_spec_reasons(result)
                record["outcome"] = "terminal"
                _atomic_json(state, state_path)
                continue
            record["last_result_state"] = fetched.state
            if int(record.get("attempt", 0)) >= 1:
                record.update(outcome="terminal", valid=False)
                _atomic_json(state, state_path)
                continue
            submit_record(candidate_id, record, retry=True)
            retried = True
        if unresolved:
            _atomic_json(state, state_path)
            raise RuntimeError("verification tasks did not reach a terminal state")
        if not retried and not any(
            record.get("outcome") == "pending" for record in records.values()
        ):
            break

    output = []
    for candidate in candidates:
        record = records[candidate["candidate_id"]]
        if record.get("outcome") != "terminal":
            raise RuntimeError("verification task inventory is not terminal")
        output.append(
            {
                "candidate_id": candidate["candidate_id"],
                "completed": True,
                "valid": bool(record.get("valid")),
                "task_id": record.get("active_id"),
                "attempt": int(record.get("attempt", 0)),
                "actual_volume_L": record.get("actual_volume_L"),
                "actual_total_loss_W": record.get("actual_total_loss_W"),
                "result": record.get("result"),
            }
        )
    document = {"schema_version": 1, "stage": stage, "results": output}
    _atomic_json(document, result_path)
    return document
