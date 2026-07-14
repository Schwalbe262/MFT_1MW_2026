"""Build offline N=3 node-local AEDT canary scheduler payloads.

This module deliberately captures the existing scheduler client's pure payload
builder without allowing any scheduler lookup or POST.  The resulting JSON is
therefore the payload that ``submit_verification`` would send, not an
approximation of its higher-level keyword arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
REGRESSION_ROOT = CAMPAIGN_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for _root in (CAMPAIGN_ROOT, REGRESSION_ROOT, VERIFY_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import attach_aware_refill_controller as controller  # noqa: E402
import pinned_pilot  # noqa: E402
import scheduler_client  # noqa: E402


CANARY_BUNDLE_ID = "mft-aedt-n3canary-260714"
CANARY_CLIENT_PREFIX = f"{CANARY_BUNDLE_ID}-client-"
EXPECTED_PROJECTS = 3
HOST_TASK_PLACEHOLDER = "{HOST_TASK_ID}"
SCHEDULER_URL_PLACEHOLDER = "{SCHEDULER_URL}"
HOST_CLONE_ROOT_PLACEHOLDER = "{HOST_CLONE_ROOT}"
DEFAULT_PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_full_action(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        isinstance(value.get("params"), Mapping)
        and isinstance(value.get("submission_env"), Mapping)
        and bool(str(value.get("workdir") or "").strip())
        and bool(str(value.get("scheduling_profile") or "").strip())
    )


def _all_full_actions(state: Mapping[str, object]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for key in ("actions", "reservations"):
        rows = state.get(key) or []
        if isinstance(rows, list):
            actions.extend(dict(row) for row in rows if _is_full_action(row))
    bundles = state.get("pooled_bundles") or []
    if isinstance(bundles, list):
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                continue
            rows = bundle.get("actions") or []
            if isinstance(rows, list):
                actions.extend(dict(row) for row in rows if _is_full_action(row))
    return actions


def _profile_from_state(state: Mapping[str, object]) -> dict[str, Any]:
    embedded = state.get("profile")
    if isinstance(embedded, Mapping):
        profile = dict(embedded)
    else:
        profile = _load_object(DEFAULT_PROFILE_PATH, "standard profile")

    generation = state.get("generation")
    identity = generation.get("identity") if isinstance(generation, Mapping) else None
    expected_hash = (
        str(identity.get("profile_sha256") or "")
        if isinstance(identity, Mapping)
        else ""
    )
    if not expected_hash:
        return profile
    if _canonical_sha256(profile) == expected_hash:
        return profile

    # The controller persists the profile hash but not the CLI timeout override.
    # Try production/operator-friendly values first, then recover any integral
    # timeout up to two days without consulting the scheduler.
    preferred = (10_800, 14_400, 7_200, 18_000, 21_600, 28_800, 43_200, 86_400)
    for timeout in preferred:
        candidate = dict(profile)
        candidate["timeout_seconds"] = timeout
        if _canonical_sha256(candidate) == expected_hash:
            return candidate
    for timeout in range(1, 172_801):
        if timeout in preferred:
            continue
        candidate = dict(profile)
        candidate["timeout_seconds"] = timeout
        if _canonical_sha256(candidate) == expected_hash:
            return candidate
    raise ValueError("state profile_sha256 cannot be reproduced from standard.json")


def _generation_identity(state: Mapping[str, object]) -> tuple[dict[str, Any], str]:
    generation = state.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("state.generation must be an object")
    identity = generation.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("state.generation.identity must be an object")
    digest = str(generation.get("digest") or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("state.generation.digest must be a SHA256")
    return dict(identity), digest


def _common_submission_env(state: Mapping[str, object]) -> dict[str, str]:
    candidates = sorted(
        _all_full_actions(state), key=lambda row: int(row.get("serial") or -1), reverse=True
    )
    for action in candidates:
        env = action.get("submission_env")
        if isinstance(env, Mapping):
            return {str(key): str(value) for key, value in env.items()}
    return {}


def _reconstruct_action(
    record: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    profile: Mapping[str, object],
    common_env: Mapping[str, str],
) -> dict[str, Any]:
    required_common_env = {
        "MFT_CONTROLLER_POLICY_SHA256",
        "MFT_DATA_CONTRACT_REVISION",
        "MFT_PROJECTS_PER_AEDT",
    }
    missing_common_env = sorted(required_common_env.difference(common_env))
    if missing_common_env:
        raise ValueError(
            "thin standalone records need a retained full action environment; "
            f"missing {missing_common_env}"
        )
    cursor_before = int(record["candidate_cursor_before"])
    expected_cursor = int(record["candidate_cursor_after"])
    expected_raw_index = int(record["candidate_raw_index"])
    seed = int(identity["candidate_seed"])
    cursor_after, raw_index, raw_params = pinned_pilot.next_valid_candidate(
        cursor_before, seed=seed
    )
    if cursor_after != expected_cursor or raw_index != expected_raw_index:
        raise ValueError(
            f"standalone serial {record.get('serial')} candidate ledger does not reproduce"
        )
    params = dict(raw_params)
    params["physics_data_revision"] = str(identity["physics_data_revision"])
    params["core_lamination_factor"] = identity["core_lamination_factor"]
    # Reservations pass through _json_copy(sort_keys=True) before submission.
    # Preserve that insertion order because verification_submission_identity
    # hashes compact JSON without sorting it a second time.
    params = dict(sorted(params.items()))
    effective = scheduler_client.effective_verification_params(params, dict(profile))
    expected_params_sha = str(record.get("params_sha256") or "")
    if expected_params_sha and _canonical_sha256(effective) != expected_params_sha:
        raise ValueError(
            f"standalone serial {record.get('serial')} parameter hash does not reproduce"
        )

    env = {str(key): str(value) for key, value in common_env.items()}
    for key in (
        "MFT_AEDT_BACKEND",
        "MFT_AEDT_SHARED_CANARY",
        "MFT_AEDT_SCHEDULER_URL",
        "MFT_SLURM_SCHEDULER_ROOT",
    ):
        env.pop(key, None)
    env.update(
        {
            "MFT_CONTROLLER_POLICY_SHA256": str(
                env["MFT_CONTROLLER_POLICY_SHA256"]
            ),
            "MFT_DATA_CONTRACT_REVISION": str(
                env["MFT_DATA_CONTRACT_REVISION"]
            ),
            "MFT_PHYSICS_DATA_REVISION": str(identity["physics_data_revision"]),
            "MFT_CORE_LAMINATION_FACTOR": str(identity["core_lamination_factor"]),
            "MFT_AEDT_BUNDLE_ID": str(record.get("bundle_id") or ""),
            "MFT_AEDT_BUNDLE_EXPECTED_ROWS": "1",
            "MFT_AEDT_BUNDLE_PROJECT_INDEX": "0",
            "MFT_PROJECTS_PER_AEDT": str(env["MFT_PROJECTS_PER_AEDT"]),
            "MFT_EXPECTED_ROWS": "1",
        }
    )
    if not env["MFT_AEDT_BUNDLE_ID"]:
        raise ValueError(
            f"standalone serial {record.get('serial')} has no source bundle ID"
        )
    name = str(record["name"])
    return {
        **dict(record),
        "workdir": str(record.get("workdir") or name.replace("-", "_")),
        "params": params,
        "submission_env": env,
        "scheduling_profile": str(
            record.get("scheduling_profile") or "fea_bursty"
        ),
    }


def select_recent_standalone_actions(
    state: Mapping[str, object], profile: Mapping[str, object]
) -> list[dict[str, Any]]:
    identity, _ = _generation_identity(state)
    full_actions = _all_full_actions(state)
    full_by_serial = {
        int(action["serial"]): action
        for action in full_actions
        if action.get("serial") is not None
    }

    submissions = [
        dict(row)
        for row in (state.get("submissions") or [])
        if isinstance(row, Mapping) and str(row.get("backend")) == "standalone"
    ]
    submissions.sort(
        key=lambda row: (
            _timestamp(row.get("submitted_or_reconciled_at")),
            int(row.get("serial") or -1),
            int(row.get("task_id") or -1),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_serials: set[int] = set()
    common_env = _common_submission_env(state)
    for record in submissions:
        if len(selected) >= EXPECTED_PROJECTS:
            break
        serial = int(record["serial"])
        if serial in seen_serials:
            continue
        full = full_by_serial.get(serial)
        action = (
            dict(full)
            if full is not None and str(full.get("backend")) == "standalone"
            else _reconstruct_action(
                record, identity=identity, profile=profile, common_env=common_env
            )
        )
        selected.append(action)
        seen_serials.add(serial)
    # A fixture or freshly initialized ledger may contain full standalone
    # actions that have not yet been committed. Use them only if the timestamped
    # submission history cannot supply three examples; old unresolved bundle
    # reservations must not displace newer committed actions.
    if len(selected) < EXPECTED_PROJECTS:
        remaining_full = [
            dict(action)
            for action in full_actions
            if str(action.get("backend")) == "standalone"
            and int(action.get("serial") or -1) not in seen_serials
        ]
        remaining_full.sort(
            key=lambda row: (
                _timestamp(
                    row.get("submitted_or_reconciled_at")
                    or row.get("updated_at")
                    or row.get("created_at")
                ),
                int(row.get("serial") or -1),
            ),
            reverse=True,
        )
        for action in remaining_full:
            if len(selected) >= EXPECTED_PROJECTS:
                break
            serial = int(action["serial"])
            selected.append(action)
            seen_serials.add(serial)
    if len(selected) < EXPECTED_PROJECTS:
        raise ValueError(
            f"state contains only {len(selected)} usable recent standalone actions; need 3"
        )
    return selected[:EXPECTED_PROJECTS]


class _CapturedResponse:
    status_code = 201

    @staticmethod
    def json() -> dict[str, int]:
        return {"task_id": 1}


def _capture_verification_payload(kwargs: Mapping[str, object]) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original_lock_check = scheduler_client.campaign_mutation_lock_is_held
    original_reconcile = scheduler_client.reconcile_task_id
    original_snapshot = scheduler_client.live_project_submission_snapshot
    original_post = scheduler_client.requests.post

    def fake_post(url: str, *, json: object, timeout: int) -> _CapturedResponse:
        if not str(url).endswith("/api/tasks") or timeout != 20:
            raise AssertionError("unexpected scheduler POST contract while capturing payload")
        if not isinstance(json, dict):
            raise AssertionError("scheduler payload is not an object")
        captured.update(json)
        return _CapturedResponse()

    try:
        scheduler_client.campaign_mutation_lock_is_held = lambda: True
        scheduler_client.reconcile_task_id = lambda *args, **kw: None
        scheduler_client.live_project_submission_snapshot = (
            lambda *args, **kw: {"project_submission_slots": 1}
        )
        scheduler_client.requests.post = fake_post
        task_id = scheduler_client._submit_verification_locked(**dict(kwargs))
    finally:
        scheduler_client.campaign_mutation_lock_is_held = original_lock_check
        scheduler_client.reconcile_task_id = original_reconcile
        scheduler_client.live_project_submission_snapshot = original_snapshot
        scheduler_client.requests.post = original_post
    if task_id != 1 or not captured:
        raise RuntimeError("failed to capture the scheduler verification payload")
    return captured


def _client_payload(
    action: Mapping[str, object],
    *,
    index: int,
    profile: Mapping[str, object],
    identity: Mapping[str, object],
    canary_scope: str,
) -> dict[str, Any]:
    submission_env = {
        str(key): str(value)
        for key, value in dict(action["submission_env"]).items()
    }
    # These are exactly the six overlays in build_node_canary_client_submission.
    submission_env.update(
        {
            "MFT_AEDT_BACKEND": "pooled",
            "MFT_AEDT_SHARED_CANARY": "1",
            "MFT_AEDT_SCHEDULER_URL": SCHEDULER_URL_PLACEHOLDER,
            "MFT_SLURM_SCHEDULER_ROOT": HOST_CLONE_ROOT_PLACEHOLDER,
            "MFT_PHYSICS_DATA_REVISION": str(identity["physics_data_revision"]),
            "MFT_CORE_LAMINATION_FACTOR": str(
                identity["core_lamination_factor"]
            ),
        }
    )
    kwargs = {
        "name": f"{CANARY_CLIENT_PREFIX}{index}",
        "workdir": str(action["workdir"]),
        "params": dict(action["params"]),
        "profile": dict(profile),
        "mem_mb": int(identity["memory_mb"]),
        "cpus": int(identity["cpus"]),
        "solver_revision": str(identity["solver_revision"]),
        "library_revision": str(identity["library_revision"]),
        "required_project_cap": int(identity["project_concurrency_target"]),
        "aedt_backend": "pooled",
        "scheduling_profile": str(action["scheduling_profile"]),
        "submission_env": submission_env,
        "dedupe_scope": canary_scope,
        "entrypoint": controller.NODE_CANARY_CLIENT_ENTRYPOINT,
        "same_node_as_task_id": 1,
        "payload_json": {
            "aedt_canary_bundle_id": CANARY_BUNDLE_ID,
            "aedt_canary_expected_projects": EXPECTED_PROJECTS,
        },
    }
    payload = _capture_verification_payload(kwargs)
    payload["same_node_as_task_id"] = HOST_TASK_PLACEHOLDER
    return payload


def build_payloads(
    state: Mapping[str, object], *, allocation_id: int, account: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    if isinstance(allocation_id, bool) or allocation_id <= 0:
        raise ValueError("allocation ID must be a positive integer")
    account = str(account).strip()
    if not account:
        raise ValueError("account must not be empty")
    profile = _profile_from_state(state)
    identity, generation_digest = _generation_identity(state)
    actions = select_recent_standalone_actions(state, profile)
    source_serials = [int(action["serial"]) for action in actions]
    canary_scope = _canonical_sha256(
        {
            "account": account,
            "allocation_id": allocation_id,
            "bundle_id": CANARY_BUNDLE_ID,
            "generation_digest": generation_digest,
        }
    )
    bundle = {
        "bundle_id": CANARY_BUNDLE_ID,
        "action_serials": source_serials,
        "host_dedupe_key": (
            f"mft-node-canary-host:{CANARY_BUNDLE_ID}:{canary_scope[:20]}"
        ),
    }
    host = controller._node_canary_host_payload(
        bundle, {"id": allocation_id, "account_name": account}
    )
    clients = [
        _client_payload(
            action,
            index=index,
            profile=profile,
            identity=identity,
            canary_scope=canary_scope,
        )
        for index, action in enumerate(actions, start=1)
    ]
    return host, clients, source_serials


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-id", required=True, type=int)
    parser.add_argument("--account", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state = _load_object(args.state, "controller state")
    host, clients, source_serials = build_payloads(
        state, allocation_id=args.allocation_id, account=args.account
    )
    args.out.mkdir(parents=True, exist_ok=True)
    host_path = args.out / "host_payload.json"
    _write_json(host_path, host)
    client_paths: list[str] = []
    for index, payload in enumerate(clients, start=1):
        path = args.out / f"client_payload_{index}.json"
        _write_json(path, payload)
        client_paths.append(str(path))
    print(
        json.dumps(
            {
                "client_payloads": client_paths,
                "host_payload": str(host_path),
                "source_standalone_serials": source_serials,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
