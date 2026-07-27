#!/usr/bin/env python3
"""One-shot corrected Rx-interface replacement cutover.

This tool preserves the two authenticated pre-gap N1=6 acquisition campaigns
exactly, except for the reviewed Rx-interface solver revision, a unique task
identity, and a 12-hour timeout.  It submits replacements before conditionally
cancelling the exact invalid legacy task set.  It never edits Scheduler source,
configuration, projects, allocations, or placement policy.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping
import urllib.parse
import urllib.request
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUNTIME_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
)
DEFAULT_OUTPUT = RUNTIME_ROOT / "rx_interface_corrected24_cutover_v1"
SCHEDULER = "http://127.0.0.1:8002"
SOLVER_REVISION = "c6a016c3a880acd632b12e52b02099cfe7b90fc5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
CANARY_TASK_ID = 97041
CANARY_NAME = "mft-goal-rx-shared-interface-canary-v1"
TIMEOUT_SECONDS = 43_200
CPUS = 8
MEMORY_MB = 65_536
MAX_WORKERS_PER_NODE = 1
PROJECT = "MFT_1MW_2026v1"
ACTIVE = {"queued", "attaching", "running"}
OLD_TASK_IDS = (
    96743,
    *range(97014, 97030),
    *range(97033, 97041),
)
AUTHORIZATION_PHRASE = (
    "AUTHORIZE EXACT CORRECTED RX INTERFACE 24-LANE CUTOVER"
)
PREFLIGHT_PREFIX = "THERMAL_RX_INTERFACE_PREFLIGHT_JSON="

CAMPAIGNS = (
    {
        "key": "cooler",
        "source": RUNTIME_ROOT
        / "n1_6_cooler_thermal_hedge_v1"
        / "batch_plan.json",
        "old_ids": tuple(range(97014, 97030)),
        "priority": 95,
        "name_prefix": "mft-goal-rxfix-lm2",
        "workdir_prefix": "mft_goal_rxfix_lm2",
    },
    {
        "key": "lastmile",
        "source": RUNTIME_ROOT
        / "n1_6_primary_temperature_lastmile_v2"
        / "batch_plan.json",
        "old_ids": tuple(range(97033, 97041)),
        "priority": 90,
        "name_prefix": "mft-goal-rxfix-txlast",
        "workdir_prefix": "mft_goal_rxfix_txlast",
    },
)


class CutoverError(RuntimeError):
    """Fail-closed cutover contract violation."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise CutoverError("payload is already sealed")
    result["payload_sha256"] = canonical_sha(result)
    return result


def validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CutoverError("sealed JSON object required")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema") != schema or observed != canonical_sha(unsigned):
        raise CutoverError(f"{schema} seal mismatch")
    return dict(value)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CutoverError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CutoverError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise CutoverError(f"refusing to replace {path}")
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def http_json(method: str, endpoint: str) -> dict[str, Any] | list[Any]:
    request = urllib.request.Request(
        SCHEDULER + endpoint,
        headers={"Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    except OSError as exc:
        raise CutoverError(f"{method} {endpoint} failed: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CutoverError(f"{method} {endpoint} returned invalid JSON") from exc
    if not isinstance(value, (dict, list)):
        raise CutoverError(f"{method} {endpoint} returned scalar JSON")
    return value


def http_text(endpoint: str) -> str:
    request = urllib.request.Request(
        SCHEDULER + endpoint,
        headers={"Accept": "text/plain"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise CutoverError(f"GET {endpoint} failed: {exc}") from exc


def validate_canary() -> dict[str, Any]:
    task = http_json("GET", f"/api/tasks/{CANARY_TASK_ID}")
    if not isinstance(task, dict):
        raise CutoverError("canary readback is not an object")
    expected = {
        "name": CANARY_NAME,
        "project": PROJECT,
        "aedt_backend": "standalone",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "priority": 100,
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    for key, expected_value in expected.items():
        if task.get(key) != expected_value:
            raise CutoverError(
                f"canary {key} drifted: {task.get(key)!r} != {expected_value!r}"
            )
    stdout = http_text(f"/api/tasks/{CANARY_TASK_ID}/stdout")
    markers = [
        json.loads(line[len(PREFLIGHT_PREFIX) :])
        for line in stdout.splitlines()
        if line.startswith(PREFLIGHT_PREFIX)
    ]
    if not markers:
        raise CutoverError("canary deterministic preflight marker is absent")
    marker = markers[-1]
    fixed = marker.get("fixed_cooling_identity") or {}
    shared = marker.get("rx_main_shared_operations") or []
    expected_fixed = {
        "schema": "mft-fixed-thermal-boundary-v1",
        "contract_sha256": (
            "20cee513ceb5bddad8071b8160875b11f37c382177f3c0039cb1c8fd335381b6"
        ),
        "fan_config": "dual",
        "fan_velocity_m_s": 1.5,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "thermal_pad_conductivity_W_mK": 0.2,
        "cooling_boundary_modified": False,
    }
    if (
        marker.get("schema") != "thermal-rx-interface-predispatch-v1"
        or marker.get("passed") is not True
        or marker.get("thermal_rx_block_interface_contract_version")
        != "thermal-rx-block-interface-coverage-v1"
        or marker.get("mesh_policy")
        != "b6-rx-block-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        or marker.get("mesh_plan_contract_version") != "thermal-mesh-plan-v7"
        or marker.get("rx_main_objects")
        != ["Rx_main_block_xn", "Rx_main_block_yp"]
        or marker.get("shared_intent_and_native_readback_passed") is not True
        or marker.get("native_operation_readback_passed") is not True
        or marker.get("direct_analyze_gate_passed") is not True
        or len(shared) != 1
        or shared[0].get("shared_region") is not True
        or shared[0].get("separate_objects") is not False
        or shared[0].get("native_separate_objects") is not False
    ):
        raise CutoverError("canary Rx-interface marker contract drifted")
    for key, expected_value in expected_fixed.items():
        if fixed.get(key) != expected_value:
            raise CutoverError(f"canary fixed cooling drifted at {key}")
    return {"task": task, "marker": marker, "stdout_sha256": hashlib.sha256(
        stdout.encode("utf-8")
    ).hexdigest()}


def prepare(output: Path) -> Path:
    from regression_260707.verify import scheduler_client
    from tools import mft_goal_targeted_symmetric_fea_batch as targeted

    if output.exists():
        raise CutoverError(f"output already exists: {output}")
    canary = validate_canary()
    campaigns: list[dict[str, Any]] = []
    all_names: set[str] = set()
    all_dedupes: set[str] = set()
    for spec in CAMPAIGNS:
        source_plan, source_root, source_profile = targeted._load_plan(
            spec["source"]
        )
        if (
            len(source_plan["lanes"]) != len(spec["old_ids"])
            or source_plan["solver_revision"]
            == SOLVER_REVISION
            or source_plan["library_revision"] != LIBRARY_REVISION
        ):
            raise CutoverError(f"{spec['key']} source campaign drifted")
        profile = copy.deepcopy(source_profile)
        profile["timeout_seconds"] = TIMEOUT_SECONDS
        lanes: list[dict[str, Any]] = []
        for lane, old_id in zip(source_plan["lanes"], spec["old_ids"]):
            rank = int(lane["rank"])
            geometry = lane["candidate"]["physical_geometry_sha256"]
            stem = geometry[:12]
            params_path = (source_root / lane["params"]["path"]).resolve(
                strict=True
            )
            params = read_json(params_path)
            if (
                float(params.get("core_center_gap_mm", -1)) != 0.0
                or float(params.get("fan_velocity", -1)) != 1.5
                or params.get("fan_config") != "dual"
                or float(params.get("core_plate_pad_t", -1)) != 2.0
                or float(params.get("wcp_pad_t", -1)) != 2.0
                or float(params.get("k_ins", -1)) != 0.2
            ):
                raise CutoverError(
                    f"{spec['key']} rank {rank} source physics drifted"
                )
            name = f"{spec['name_prefix']}-r{rank:02d}-{stem}-v1"
            workdir = f"{spec['workdir_prefix']}_r{rank:02d}_{stem}_v1"
            identity = scheduler_client.verification_submission_identity(
                name,
                params,
                profile,
                SOLVER_REVISION,
                LIBRARY_REVISION,
            )
            if name in all_names or identity["dedupe_key"] in all_dedupes:
                raise CutoverError("replacement identity collision")
            all_names.add(name)
            all_dedupes.add(identity["dedupe_key"])
            lanes.append(
                {
                    "rank": rank,
                    "old_task_id": old_id,
                    "physical_geometry_sha256": geometry,
                    "source_params": {
                        "path": str(params_path),
                        "sha256": file_sha(params_path),
                        "size_bytes": params_path.stat().st_size,
                    },
                    "name": name,
                    "workdir": workdir,
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": canonical_sha(
                        identity["merged"]
                    ),
                    "resources": {
                        "project": PROJECT,
                        "cpus": CPUS,
                        "memory_mb": MEMORY_MB,
                        "gpus": 0,
                        "priority": spec["priority"],
                        "timeout_seconds": TIMEOUT_SECONDS,
                        "max_workers_per_node": MAX_WORKERS_PER_NODE,
                        "aedt_backend": "standalone",
                    },
                }
            )
        campaign_plan = seal(
            {
                "schema": "mft-corrected-rx-interface-replacement-plan-v1",
                "created_at_utc": now(),
                "campaign": spec["key"],
                "classification": (
                    "pre-gap-symmetric-unrounded-acquisition-only;"
                    "not-final-Lm2mH-truth"
                ),
                "source_plan": {
                    "path": str(spec["source"]),
                    "sha256": file_sha(spec["source"]),
                    "payload_sha256": source_plan["payload_sha256"],
                },
                "source_solver_revision": source_plan["solver_revision"],
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "canary_task_id": CANARY_TASK_ID,
                "canary_marker_sha256": canonical_sha(canary["marker"]),
                "profile": profile,
                "profile_sha256": canonical_sha(profile),
                "scheduler_repository_modified": False,
                "scheduler_configuration_modified": False,
                "submission_performed": False,
                "lanes": lanes,
            }
        )
        campaigns.append(campaign_plan)
    if sum(len(item["lanes"]) for item in campaigns) != 24:
        raise CutoverError("replacement count is not exactly 24")
    output.mkdir(parents=True)
    try:
        for campaign in campaigns:
            write_json(output / f"{campaign['campaign']}_plan.json", campaign)
        aggregate = seal(
            {
                "schema": "mft-corrected-rx-interface-cutover-plan-v1",
                "created_at_utc": now(),
                "canary": canary,
                "replacement_count": 24,
                "old_task_ids": list(OLD_TASK_IDS),
                "campaigns": [
                    {
                        "key": campaign["campaign"],
                        "path": f"{campaign['campaign']}_plan.json",
                        "sha256": file_sha(
                            output / f"{campaign['campaign']}_plan.json"
                        ),
                        "payload_sha256": campaign["payload_sha256"],
                    }
                    for campaign in campaigns
                ],
                "submit_before_cancel": True,
                "conditional_cancel_statuses": sorted(ACTIVE),
                "scheduler_repository_modified": False,
                "scheduler_configuration_modified": False,
            }
        )
        write_json(output / "cutover_plan.json", aggregate)
    except BaseException:
        raise
    return output / "cutover_plan.json"


def authorize(plan_path: Path, output: Path, authorized_by: str) -> Path:
    plan = validate_seal(
        read_json(plan_path),
        "mft-corrected-rx-interface-cutover-plan-v1",
    )
    if not authorized_by.strip():
        raise CutoverError("authorized_by is required")
    receipt = seal(
        {
            "schema": "mft-corrected-rx-interface-cutover-authorization-v1",
            "authorized": True,
            "authorized_at_utc": now(),
            "authorized_by": authorized_by,
            "confirmation_phrase": AUTHORIZATION_PHRASE,
            "cutover_plan_sha256": file_sha(plan_path),
            "cutover_plan_payload_sha256": plan["payload_sha256"],
            "replacement_count": 24,
            "old_task_ids": list(OLD_TASK_IDS),
        }
    )
    write_json(output, receipt)
    return output


def load_bundle(
    plan_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    aggregate = validate_seal(
        read_json(plan_path),
        "mft-corrected-rx-interface-cutover-plan-v1",
    )
    authorization = validate_seal(
        read_json(authorization_path),
        "mft-corrected-rx-interface-cutover-authorization-v1",
    )
    if (
        authorization.get("authorized") is not True
        or authorization.get("confirmation_phrase") != AUTHORIZATION_PHRASE
        or authorization.get("cutover_plan_sha256") != file_sha(plan_path)
        or authorization.get("cutover_plan_payload_sha256")
        != aggregate["payload_sha256"]
        or authorization.get("old_task_ids") != list(OLD_TASK_IDS)
    ):
        raise CutoverError("authorization receipt drifted")
    campaigns = []
    for record in aggregate["campaigns"]:
        path = (plan_path.parent / record["path"]).resolve(strict=True)
        campaign = validate_seal(
            read_json(path),
            "mft-corrected-rx-interface-replacement-plan-v1",
        )
        if (
            file_sha(path) != record["sha256"]
            or campaign["payload_sha256"] != record["payload_sha256"]
        ):
            raise CutoverError("campaign plan bytes drifted")
        campaigns.append(campaign)
    if sum(len(item["lanes"]) for item in campaigns) != 24:
        raise CutoverError("replacement bundle count drifted")
    return aggregate, authorization, campaigns


def submit(
    plan_path: Path, authorization_path: Path, output: Path
) -> Path:
    from regression_260707.verify import scheduler_client
    from tools import mft_goal_targeted_symmetric_fea_batch as targeted

    aggregate, authorization, campaigns = load_bundle(
        plan_path, authorization_path
    )
    canary = validate_canary()
    if canary["marker"] != aggregate["canary"]["marker"]:
        raise CutoverError("canary marker changed since plan preparation")
    if output.exists():
        raise CutoverError(f"submission receipt already exists: {output}")
    attempt = seal(
        {
            "schema": "mft-corrected-rx-interface-cutover-attempt-v1",
            "created_at_utc": now(),
            "cutover_plan_sha256": file_sha(plan_path),
            "authorization_sha256": file_sha(authorization_path),
            "maximum_scheduler_posts": 24,
            "post_budget_consumed": True,
        }
    )
    write_json(output.parent / "submission_attempt.json", attempt)
    submissions: list[dict[str, Any]] = []
    progress_path = output.parent / "submission_progress.json"
    lane_queue: list[tuple[dict[str, Any], dict[str, Any]]] = []
    longest = max(len(campaign["lanes"]) for campaign in campaigns)
    # Interleave the p95 cooler and p90 last-mile plans.  The first two POSTs
    # therefore start one authenticated lane from each campaign immediately.
    for index in range(longest):
        for campaign in campaigns:
            if index < len(campaign["lanes"]):
                lane_queue.append((campaign, campaign["lanes"][index]))
    for campaign, lane in lane_queue:
        profile = campaign["profile"]
        priority = campaign["lanes"][0]["resources"]["priority"]
        params_path = Path(lane["source_params"]["path"])
        if (
            file_sha(params_path) != lane["source_params"]["sha256"]
            or params_path.stat().st_size
            != lane["source_params"]["size_bytes"]
        ):
            raise CutoverError("source parameter bytes drifted")
        params = read_json(params_path)
        identity = scheduler_client.verification_submission_identity(
            lane["name"],
            params,
            profile,
            SOLVER_REVISION,
            LIBRARY_REVISION,
        )
        if (
            identity["dedupe_key"] != lane["dedupe_key"]
            or identity["parameter_digest"] != lane["parameter_digest"]
            or canonical_sha(identity["merged"])
            != lane["effective_params_sha256"]
        ):
            raise CutoverError("replacement execution identity drifted")
        evidence = scheduler_client.submit_verification(
            lane["name"],
            lane["workdir"],
            params,
            profile,
            mem_mb=MEMORY_MB,
            cpus=CPUS,
            solver_revision=SOLVER_REVISION,
            library_revision=LIBRARY_REVISION,
            priority=priority,
            max_workers_per_node=MAX_WORKERS_PER_NODE,
            aedt_backend="standalone",
            submission_env=targeted._core_environment(SOLVER_REVISION),
            required_hard_cap=targeted.PROJECT_ACTIVE_TASK_CAP,
            return_submission_evidence=True,
            max_project_active_tasks=targeted.PROJECT_ACTIVE_TASK_CAP,
            scheduler_url=SCHEDULER,
        )
        if not isinstance(evidence, dict):
            raise CutoverError("submission evidence is absent")
        task_id = int(evidence.get("task_id") or 0)
        readback = http_json("GET", f"/api/tasks/{task_id}")
        if not isinstance(readback, dict):
            raise CutoverError("replacement GET is not an object")
        resources = lane["resources"]
        expected = {
            "name": lane["name"],
            "dedupe_key": lane["dedupe_key"],
            "project": PROJECT,
            "aedt_backend": "standalone",
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "gpus": 0,
            "priority": resources["priority"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_workers_per_node": MAX_WORKERS_PER_NODE,
        }
        for key, expected_value in expected.items():
            if readback.get(key) != expected_value:
                raise CutoverError(
                    f"task {task_id} readback drifted at {key}"
                )
        if readback.get("status") not in ACTIVE:
            raise CutoverError(
                f"task {task_id} is not accepted-active: "
                f"{readback.get('status')}"
            )
        submissions.append(
            {
                "campaign": campaign["campaign"],
                "rank": lane["rank"],
                "old_task_id": lane["old_task_id"],
                "physical_geometry_sha256": lane[
                    "physical_geometry_sha256"
                ],
                "task_id": task_id,
                "name": lane["name"],
                "dedupe_key": lane["dedupe_key"],
                "scheduler_post_performed": evidence.get(
                    "scheduler_mutation_performed"
                )
                is True,
                "evidence": evidence,
                "readback": readback,
                "readback_sha256": canonical_sha(readback),
            }
        )
        write_json(
            progress_path,
            {
                "schema": (
                    "mft-corrected-rx-interface-cutover-progress-v1"
                ),
                "updated_at_utc": now(),
                "expected_count": 24,
                "submitted_count": len(submissions),
                "submissions": submissions,
            },
            replace=True,
        )
    if (
        len(submissions) != 24
        or len({row["task_id"] for row in submissions}) != 24
        or any(
            row["scheduler_post_performed"] is not True for row in submissions
        )
    ):
        raise CutoverError("24 unique mutating POST receipts are not complete")
    receipt = seal(
        {
            "schema": "mft-corrected-rx-interface-submission-receipt-v1",
            "created_at_utc": now(),
            "cutover_plan_sha256": file_sha(plan_path),
            "authorization_sha256": file_sha(authorization_path),
            "canary_task_id": CANARY_TASK_ID,
            "canary_marker_sha256": canonical_sha(canary["marker"]),
            "scheduler_post_calls": 24,
            "replacement_count": 24,
            "all_get_identities_valid": True,
            "all_tasks_accepted_active": True,
            "submissions": submissions,
        }
    )
    write_json(output, receipt)
    return output


def verify_and_cancel(
    plan_path: Path,
    authorization_path: Path,
    submission_path: Path,
    output: Path,
) -> Path:
    aggregate, _authorization, campaigns = load_bundle(
        plan_path, authorization_path
    )
    receipt = validate_seal(
        read_json(submission_path),
        "mft-corrected-rx-interface-submission-receipt-v1",
    )
    if (
        receipt.get("replacement_count") != 24
        or receipt.get("scheduler_post_calls") != 24
        or receipt.get("all_get_identities_valid") is not True
        or receipt.get("all_tasks_accepted_active") is not True
    ):
        raise CutoverError("submission receipt is incomplete")
    expected_lanes = {
        lane["name"]: lane for campaign in campaigns for lane in campaign["lanes"]
    }
    before_new = []
    for row in receipt["submissions"]:
        task = http_json("GET", f"/api/tasks/{row['task_id']}")
        if not isinstance(task, dict):
            raise CutoverError("replacement re-read is not an object")
        lane = expected_lanes.get(row["name"])
        if lane is None:
            raise CutoverError("replacement name is outside sealed plans")
        resources = lane["resources"]
        if (
            task.get("name") != lane["name"]
            or task.get("dedupe_key") != lane["dedupe_key"]
            or task.get("project") != PROJECT
            or task.get("aedt_backend") != "standalone"
            or task.get("cpus") != CPUS
            or task.get("memory_mb") != MEMORY_MB
            or task.get("priority") != resources["priority"]
            or task.get("timeout_seconds") != TIMEOUT_SECONDS
            or task.get("status") not in ACTIVE
        ):
            raise CutoverError(
                f"replacement task {row['task_id']} failed final GET gate"
            )
        before_new.append(task)
    old_before = []
    for task_id in OLD_TASK_IDS:
        task = http_json("GET", f"/api/tasks/{task_id}")
        if not isinstance(task, dict):
            raise CutoverError("old task readback is not an object")
        old_before.append(task)
    query = urllib.parse.urlencode(
        {
            "task_ids": ",".join(str(value) for value in OLD_TASK_IDS),
            "statuses": "queued,attaching,running",
        }
    )
    cancel_response = http_json(
        "POST", f"/api/tasks/cancel?{query}"
    )
    old_after = []
    for task_id in OLD_TASK_IDS:
        task = http_json("GET", f"/api/tasks/{task_id}")
        if not isinstance(task, dict):
            raise CutoverError("old post-cancel readback is not an object")
        old_after.append(task)
    new_after = []
    for task in before_new:
        readback = http_json("GET", f"/api/tasks/{task['task_id']}")
        if not isinstance(readback, dict):
            raise CutoverError("new post-cancel readback is not an object")
        if readback.get("status") not in ACTIVE:
            raise CutoverError(
                f"replacement {readback.get('task_id')} lost active status"
            )
        new_after.append(readback)
    value = seal(
        {
            "schema": "mft-corrected-rx-interface-cutover-receipt-v1",
            "created_at_utc": now(),
            "cutover_plan_payload_sha256": aggregate["payload_sha256"],
            "submission_receipt_sha256": file_sha(submission_path),
            "old_task_ids": list(OLD_TASK_IDS),
            "conditional_cancel_statuses": sorted(ACTIVE),
            "old_before": old_before,
            "cancel_response": cancel_response,
            "old_after": old_after,
            "new_before": before_new,
            "new_after": new_after,
            "replacement_compute_gap_created": False,
            "scheduler_repository_modified": False,
            "scheduler_configuration_modified": False,
        }
    )
    write_json(output, value)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    auth_command = commands.add_parser("authorize")
    auth_command.add_argument("--plan", type=Path, required=True)
    auth_command.add_argument("--output", type=Path, required=True)
    auth_command.add_argument("--authorized-by", required=True)
    submit_command = commands.add_parser("submit")
    submit_command.add_argument("--plan", type=Path, required=True)
    submit_command.add_argument("--authorization", type=Path, required=True)
    submit_command.add_argument("--output", type=Path, required=True)
    submit_command.add_argument("--apply", action="store_true")
    cancel_command = commands.add_parser("verify-and-cancel")
    cancel_command.add_argument("--plan", type=Path, required=True)
    cancel_command.add_argument("--authorization", type=Path, required=True)
    cancel_command.add_argument("--submission", type=Path, required=True)
    cancel_command.add_argument("--output", type=Path, required=True)
    cancel_command.add_argument("--apply", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        path = prepare(args.output.resolve())
    elif args.command == "authorize":
        path = authorize(
            args.plan.resolve(strict=True),
            args.output.resolve(),
            args.authorized_by,
        )
    elif args.command == "submit":
        if not args.apply:
            raise CutoverError("submit requires --apply")
        path = submit(
            args.plan.resolve(strict=True),
            args.authorization.resolve(strict=True),
            args.output.resolve(),
        )
    else:
        if not args.apply:
            raise CutoverError("verify-and-cancel requires --apply")
        path = verify_and_cancel(
            args.plan.resolve(strict=True),
            args.authorization.resolve(strict=True),
            args.submission.resolve(strict=True),
            args.output.resolve(),
        )
    print(json.dumps({"status": "ok", "path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CutoverError as exc:
        print(f"CUTOVER_REFUSED: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
