"""Submit and collect a bounded turn-graded terminal-capacitance sweep.

The source is one authenticated physical-gap tuning lane.  Each Scheduler task
re-solves the symmetric magnetic matrix and exactly one electrostatic winding
with prescribed per-turn voltages.  Variants bound section order, radial
direction, terminal polarity, side-section polarity, and midpoint/endpoint
voltage conventions.  The symmetric result is explicitly diagnostic because
its electrostatic cut planes assume even potential and cannot attest an
unknown full-model series interconnect.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_lm2mh_gap_tuner as gap_tuner  # noqa: E402


PLAN_SCHEMA = "mft-goal-turn-graded-cap-batch-plan-v1"
SUBMISSION_SCHEMA = "mft-goal-turn-graded-cap-submission-v1"
COLLECTION_SCHEMA = "mft-goal-turn-graded-cap-collection-v1"
PROFILE_SCHEMA = "mft-goal-turn-graded-cap-profile-v1"
SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_CAMPAIGN = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rank1_neighborhood_2b213_physical_gap_tuning_v1"
    r"\campaign_manifest.json"
)
DEFAULT_ROUND = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rank1_neighborhood_2b213_physical_gap_tuning_v1"
    r"\round-01\round_plan.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rank1_turn_graded_cap_sweep_v1"
)
ELECTROSTATIC_CAP_SOURCE = REPOSITORY_ROOT / "module" / "electrostatic_cap.py"
INPUT_PARAMETER_SOURCE = (
    REPOSITORY_ROOT / "module" / "input_parameter_260706.py"
)
RUN_SIMULATION_SOURCE = REPOSITORY_ROOT / "run_simulation_260706.py"
EXPECTED_ACTUAL_TOPOLOGY_SOURCE_SHA256 = {
    "module/electrostatic_cap.py": (
        "8b383584cdc62310f98283afe6f12b1036fc46fdb8a00c4b4c8b9ab3f71454b1"
    ),
    "module/input_parameter_260706.py": (
        "761861d51dd2f4f12219a6b9277c14c7574a1d03303b42ec453a30941ac11625"
    ),
    "run_simulation_260706.py": (
        "b0dcc5fd38883ceec9cd754d9cae2a8750f98e1093695cd5227777e69bc2a631"
    ),
}
ACTUAL_CONNECTION_VARIANT_IDS = {
    "Tx": "tx-main-mid",
    "Rx": "rx-main-side-mid",
}
OPPOSED_SIDE_SENSITIVITY_VARIANT_ID = "rx-main-side-mid-opposed-side"
RESONANCE_MIN_HZ = 15_000.0
CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 7_200
PROJECT_ACTIVE_TASK_CAP = 500
HEX40 = re.compile(r"[0-9a-f]{40}")


class TurnGradedBatchError(RuntimeError):
    """The diagnostic source, submission, or result contract drifted."""


VARIANTS = (
    {
        "id": "tx-main-mid",
        "active": "Tx",
        "order": "auto",
        "reverse": "none",
        "terminal_reverse": 0,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "tx-main-mid-terminal-reverse",
        "active": "Tx",
        "order": "auto",
        "reverse": "none",
        "terminal_reverse": 1,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-main-side-mid",
        "active": "Rx",
        "order": "main,side",
        "reverse": "none",
        "terminal_reverse": 0,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-main-side-mid-terminal-reverse",
        "active": "Rx",
        "order": "main,side",
        "reverse": "none",
        "terminal_reverse": 1,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-side-main-mid",
        "active": "Rx",
        "order": "side,main",
        "reverse": "none",
        "terminal_reverse": 0,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-side-main-mid-terminal-reverse",
        "active": "Rx",
        "order": "side,main",
        "reverse": "none",
        "terminal_reverse": 1,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-main-side-mid-reverse-side",
        "active": "Rx",
        "order": "main,side",
        "reverse": "side",
        "terminal_reverse": 0,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-main-side-mid-reverse-both",
        "active": "Rx",
        "order": "main,side",
        "reverse": "main,side",
        "terminal_reverse": 0,
        "side_polarity": 1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-main-side-mid-opposed-side",
        "active": "Rx",
        "order": "main,side",
        "reverse": "none",
        "terminal_reverse": 0,
        "side_polarity": -1,
        "policy": "turn_midpoint",
    },
    {
        "id": "rx-main-side-endpoints",
        "active": "Rx",
        "order": "main,side",
        "reverse": "none",
        "terminal_reverse": 0,
        "side_polarity": 1,
        "policy": "terminal_endpoints",
    },
)


def _sha(value: Any) -> str:
    return gap_tuner._sha(value)  # noqa: SLF001


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    return gap_tuner._seal(value)  # noqa: SLF001


def _read(path: Path) -> dict[str, Any]:
    return gap_tuner._read_json(path)  # noqa: SLF001


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return gap_tuner._file_record(path, relative_to=relative_to)  # noqa: SLF001


def _actual_connection_topology_evidence(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the current solver's intended additive-series topology basis.

    Maxwell coil-sheet polarity follows the selected sheet-face orientation,
    so its Positive/Negative labels are not a separate electrical section
    polarity.  The turn-graded terminal schedule therefore uses the explicit
    electrical convention in ``linear_turn_voltage_schedule``: every section
    defaults to +1 and the production input keeps both side sections at +1.
    """

    sources = {
        "module/electrostatic_cap.py": ELECTROSTATIC_CAP_SOURCE,
        "module/input_parameter_260706.py": INPUT_PARAMETER_SOURCE,
        "run_simulation_260706.py": RUN_SIMULATION_SOURCE,
    }
    records = {
        name: _record(path)
        for name, path in sources.items()
    }
    for name, expected in EXPECTED_ACTUAL_TOPOLOGY_SOURCE_SHA256.items():
        if records[name]["sha256"] != expected:
            raise TurnGradedBatchError(
                f"actual-connection topology source drifted: {name}"
            )
    electrostatic_text = ELECTROSTATIC_CAP_SOURCE.read_text(encoding="utf-8")
    input_text = INPUT_PARAMETER_SOURCE.read_text(encoding="utf-8")
    runner_text = RUN_SIMULATION_SOURCE.read_text(encoding="utf-8")
    required_fragments = {
        "electrostatic_default_all_requested_sections_additive": (
            electrostatic_text,
            "{section: 1 for section in requested_sections}",
        ),
        "input_default_retained_side_additive": (
            input_text,
            '"cap_turn_graded_side_polarity": 1',
        ),
        "input_default_mirrored_side2_additive": (
            input_text,
            '"cap_turn_graded_side2_polarity": 1',
        ),
        "runtime_main_schedule_additive": (
            runner_text,
            'section_polarities = {"main": 1}',
        ),
        "runtime_side_schedule_from_sealed_input": (
            runner_text,
            'section_polarities["side"] = int(',
        ),
        "matrix_main_and_side_share_one_rx_winding": (
            runner_text,
            'add_winding_coils(assignment="Rx_winding", '
            "coils=[coil.name for coil in self.Rx_coil])",
        ),
        "retained_main_inward_face_positive": (
            runner_text,
            'polarity="Positive", name=f"Rx_center_coil_in_{idx}"',
        ),
        "retained_side_inward_face_positive": (
            runner_text,
            'polarity="Positive", name=f"Rx_side_coil_in_{idx}"',
        ),
    }
    missing = [
        name
        for name, (text, fragment) in required_fragments.items()
        if fragment not in text
    ]
    if missing:
        raise TurnGradedBatchError(
            "actual-connection topology code evidence is absent: "
            + ",".join(missing)
        )
    selected_ids = set(plan.get("selected_variant_ids") or [])
    required_variants = {
        *ACTUAL_CONNECTION_VARIANT_IDS.values(),
        OPPOSED_SIDE_SENSITIVITY_VARIANT_ID,
    }
    if not required_variants.issubset(selected_ids):
        raise TurnGradedBatchError(
            "actual/opposed topology variants are absent from the sweep"
        )
    return {
        "sealed": True,
        "basis": (
            "current symmetric solver intended additive-series electrical "
            "connection proxy; Maxwell sheet polarity labels compensate face "
            "orientation and do not negate the section voltage schedule"
        ),
        "solver_revision_used_by_sweep": plan["solver_revision"],
        "actual_connection_variant_ids": dict(
            ACTUAL_CONNECTION_VARIANT_IDS
        ),
        "actual_Rx_section_schedule": {
            "section_order": ["main", "side"],
            "section_polarities": {"main": 1, "side": 1},
            "reverse_sections": [],
            "reverse_terminal_polarity": False,
            "voltage_policy": "turn_midpoint",
        },
        "opposed_side_variant": {
            "variant_id": OPPOSED_SIDE_SENSITIVITY_VARIANT_ID,
            "section_polarities": {"main": 1, "side": -1},
            "classification": "sensitivity-only-not-actual-design",
        },
        "source_files": records,
        "code_assertions": {
            name: fragment
            for name, (_text, fragment) in required_fragments.items()
        },
        "full_model_series_interconnect_attested": False,
        "limitation": (
            "This seals the intended connection basis implemented by the "
            "current symmetric solver; a retained symmetric electrostatic "
            "model still does not independently attest the future full-model "
            "physical terminal interconnect."
        ),
    }


def _write(path: Path, value: Any) -> Path:
    return gap_tuner._atomic_json(path, value)  # noqa: SLF001


def _core_environment(solver_revision: str) -> dict[str, str]:
    contract = "mft-standalone-core-optin-v1"
    auth = _sha(
        {
            "backend": "standalone",
            "contract_version": contract,
            "requested_num_cores": CPUS,
            "required_slurm_cpus_per_task": CPUS,
            "solver_revision": solver_revision,
        }
    )
    return {
        "MFT_STANDALONE_CORE_CONTRACT": contract,
        "MFT_STANDALONE_CORE_COUNT": str(CPUS),
        "MFT_STANDALONE_CORE_AUTH_SHA256": auth,
    }


def _profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA,
        "stage": "diagnostic-turn-graded-capacitance",
        "comment": (
            "Symmetric non-rounded matrix plus one turn-graded "
            "electrostatic active winding"
        ),
        "reviewed_solver_path": "run_simulation_260706.py --fixed --headless",
        "cli_flags": "--headless",
        "param_overrides": {
            "full_model": 0,
            "round_corner": 0,
            "matrix_on": 1,
            "cap_on": 0,
            "loss_on": 0,
            "thermal_on": 0,
            "matrix_skin_mesh": 0,
            "matrix_percent_error": 0.5,
            "matrix_max_passes": 30,
            "matrix_min_converged": 1,
            "cap_percent_error": 1.0,
            "cap_max_passes": 10,
            "keep_project": 0,
        },
        "mem_mb": MEMORY_MB,
        "cpus": CPUS,
        "timeout_seconds": TIMEOUT_SECONDS,
    }


def prepare(
    *,
    campaign_path: Path,
    round_path: Path,
    output: Path,
    solver_revision: str,
    library_revision: str,
    variant_ids: tuple[str, ...] | None = None,
) -> Path:
    solver = str(solver_revision).lower()
    library = str(library_revision).lower()
    if not HEX40.fullmatch(solver) or not HEX40.fullmatch(library):
        raise TurnGradedBatchError("full solver/library revisions required")
    campaign, campaign_root = gap_tuner._load_campaign(  # noqa: SLF001
        campaign_path
    )
    round_plan, round_root = gap_tuner._load_round(  # noqa: SLF001
        round_path, campaign, campaign_root
    )
    if len(round_plan["lanes"]) != 1:
        raise TurnGradedBatchError("exactly one physical-gap lane is required")
    source_lane = round_plan["lanes"][0]
    source_params_path = (
        round_root / source_lane["params"]["path"]
    ).resolve(strict=True)
    source = _read(source_params_path)
    gap_mm = float(source["core_center_gap_mm"])
    if not math.isfinite(gap_mm) or gap_mm <= 0.0:
        raise TurnGradedBatchError("positive physical center gap is required")
    base = create_input_parameter(source).iloc[0].to_dict()
    profile = _profile()
    destination = output.resolve()
    if destination.exists():
        raise TurnGradedBatchError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    profile_path = _write(destination / "profile.json", profile)
    variant_by_id = {variant["id"]: variant for variant in VARIANTS}
    requested_ids = tuple(variant_ids or variant_by_id)
    if (
        not requested_ids
        or len(requested_ids) != len(set(requested_ids))
        or any(item not in variant_by_id for item in requested_ids)
    ):
        raise TurnGradedBatchError(
            "variant_ids must be a non-empty unique subset of known variants"
        )
    selected_variants = tuple(variant_by_id[item] for item in requested_ids)
    lanes = []
    for index, variant in enumerate(selected_variants, start=1):
        params = {
            key: copy.deepcopy(base[key])
            for key in sorted(ALL_INPUT_KEYS)
        }
        params.update(profile["param_overrides"])
        params.update(
            {
                "cap_turn_graded_active_winding": variant["active"],
                "cap_turn_graded_voltage_policy": variant["policy"],
                "cap_turn_graded_section_order": variant["order"],
                "cap_turn_graded_reverse_sections": variant["reverse"],
                "cap_turn_graded_reverse_terminal_polarity": variant[
                    "terminal_reverse"
                ],
                "cap_turn_graded_side_polarity": variant["side_polarity"],
                "cap_turn_graded_side2_polarity": 1,
            }
        )
        validation_check(create_input_parameter(params), strict=True)
        token = variant["id"].replace("-", "_")
        params_path = _write(
            destination / "params" / f"{index:02d}-{token}.json",
            params,
        )
        name = f"mft-capgraded-{index:02d}-{variant['id']}"
        identity = scheduler_client.verification_submission_identity(
            name, params, profile, solver, library
        )
        lanes.append(
            {
                "lane_index": index,
                "variant": copy.deepcopy(variant),
                "params": _record(params_path, relative_to=destination),
                "params_sha256": _sha(params),
                "scheduler": {
                    "name": name,
                    "workdir": f"mft_capgraded_{index:02d}_{token}",
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": _sha(identity["merged"]),
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "timeout_seconds": TIMEOUT_SECONDS,
                    "priority": 98,
                    "max_workers_per_node": 1,
                    "environment": _core_environment(solver),
                },
            }
        )
    plan = _seal(
        {
            "schema_version": PLAN_SCHEMA,
            "solver_revision": solver,
            "library_revision": library,
            "source": {
                "campaign": _record(campaign_path),
                "campaign_payload_sha256": campaign["payload_sha256"],
                "round": _record(round_path),
                "round_payload_sha256": round_plan["payload_sha256"],
                "params": _record(source_params_path),
                "params_payload_sha256": _sha(source),
                "core_center_gap_mm": gap_mm,
                "physical_geometry_sha256": campaign["candidate"][
                    "physical_geometry_sha256"
                ],
                "gap_status_at_prepare": (
                    "interpolated_gap_under_parallel_direct_matrix_validation"
                ),
            },
            "profile": _record(profile_path, relative_to=destination),
            "profile_sha256": _sha(profile),
            "lanes": lanes,
            "lane_count": len(lanes),
            "selected_variant_ids": list(requested_ids),
            "parallel_execution_requested": True,
            "symmetric_nonrounded": True,
            "full_model_series_interconnect_attested": False,
            "final_design_pass_allowed": False,
        }
    )
    return _write(destination / "batch_plan.json", plan)


def _load_plan(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan_path = path.resolve(strict=True)
    root = plan_path.parent
    plan = gap_tuner._validate_seal(  # noqa: SLF001
        _read(plan_path), PLAN_SCHEMA
    )
    profile_path = (root / plan["profile"]["path"]).resolve(strict=True)
    profile = _read(profile_path)
    variant_by_id = {variant["id"]: variant for variant in VARIANTS}
    selected_ids = plan.get("selected_variant_ids")
    selected_variants = (
        [variant_by_id.get(str(item)) for item in selected_ids]
        if isinstance(selected_ids, list)
        else []
    )
    if (
        not selected_variants
        or any(variant is None for variant in selected_variants)
        or len(selected_ids) != len(set(selected_ids))
        or plan.get("lane_count") != len(selected_variants)
        or len(plan.get("lanes") or []) != len(selected_variants)
        or [
            lane.get("variant") for lane in plan.get("lanes") or []
        ] != selected_variants
        or plan.get("parallel_execution_requested") is not True
        or plan.get("symmetric_nonrounded") is not True
        or plan.get("final_design_pass_allowed") is not False
        or _record(profile_path, relative_to=root) != plan["profile"]
        or _sha(profile) != plan["profile_sha256"]
        or profile != _profile()
    ):
        raise TurnGradedBatchError("batch plan contract drifted")
    identities: set[str] = set()
    for lane in plan["lanes"]:
        params_path = (root / lane["params"]["path"]).resolve(strict=True)
        params = _read(params_path)
        scheduler = lane["scheduler"]
        identity = scheduler_client.verification_submission_identity(
            scheduler["name"],
            params,
            profile,
            plan["solver_revision"],
            plan["library_revision"],
        )
        if (
            _record(params_path, relative_to=root) != lane["params"]
            or _sha(params) != lane["params_sha256"]
            or identity["dedupe_key"] != scheduler["dedupe_key"]
            or identity["parameter_digest"] != scheduler["parameter_digest"]
            or _sha(identity["merged"])
            != scheduler["effective_params_sha256"]
            or scheduler["dedupe_key"] in identities
        ):
            raise TurnGradedBatchError("lane identity drifted")
        identities.add(scheduler["dedupe_key"])
    return plan, root, profile


def submit(
    *,
    plan_path: Path,
    output: Path,
    scheduler_url: str,
) -> Path:
    plan, root, profile = _load_plan(plan_path)
    submitted = []
    try:
        for lane in plan["lanes"]:
            params = _read(root / lane["params"]["path"])
            scheduler = lane["scheduler"]
            evidence = scheduler_client.submit_verification(
                scheduler["name"],
                scheduler["workdir"],
                params,
                profile,
                mem_mb=scheduler["memory_mb"],
                cpus=scheduler["cpus"],
                solver_revision=plan["solver_revision"],
                library_revision=plan["library_revision"],
                priority=scheduler["priority"],
                max_workers_per_node=scheduler["max_workers_per_node"],
                aedt_backend="standalone",
                submission_env=scheduler["environment"],
                required_hard_cap=PROJECT_ACTIVE_TASK_CAP,
                max_project_active_tasks=PROJECT_ACTIVE_TASK_CAP,
                return_submission_evidence=True,
                scheduler_url=scheduler_url,
            )
            if not isinstance(evidence, dict):
                raise TurnGradedBatchError("submission evidence is absent")
            readback = (
                evidence.get("api_pre_submission_readback")
                or evidence.get("api_post_submission_response")
                or evidence
            )
            if isinstance(readback, dict) and isinstance(
                    readback.get("task"), dict):
                readback = readback["task"]
            task_id = int(
                evidence.get("task_id")
                or (readback or {}).get("id")
                or (readback or {}).get("task_id")
                or 0
            )
            if (
                not isinstance(readback, dict)
                or task_id <= 0
                or readback.get("name") != scheduler["name"]
                or readback.get("dedupe_key") != scheduler["dedupe_key"]
                or int(readback.get("cpus") or 0) != CPUS
                or int(readback.get("memory_mb") or 0) != MEMORY_MB
            ):
                raise TurnGradedBatchError("submission readback drifted")
            submitted.append(
                {
                    "lane_index": lane["lane_index"],
                    "variant": lane["variant"],
                    "task_id": task_id,
                    "name": scheduler["name"],
                    "dedupe_key": scheduler["dedupe_key"],
                    "readback": readback,
                    "submission_evidence": evidence,
                }
            )
    except BaseException as error:
        _write(
            output,
            _seal(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "plan": _record(plan_path),
                    "plan_payload_sha256": plan["payload_sha256"],
                    "submissions": submitted,
                    "complete": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
            ),
        )
        raise
    return _write(
        output,
        _seal(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "plan": _record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "scheduler_url": scheduler_url.rstrip("/"),
                "submissions": submitted,
                "submitted_lane_count": len(submitted),
                "complete": len(submitted) == len(plan["lanes"]),
                "parallel_execution_requested": True,
            }
        ),
    )


def _api_task(base: str, task_id: int) -> dict[str, Any]:
    url = (
        f"{base.rstrip('/')}/api/tasks/{task_id}"
        "?include_output=true&output_limit=100000"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TurnGradedBatchError("Scheduler task response is not an object")
    task = value.get("task", value)
    if not isinstance(task, dict):
        raise TurnGradedBatchError("Scheduler task readback is absent")
    return task


def _result(task: Mapping[str, Any]) -> dict[str, Any] | None:
    value = task.get("result_json")
    if isinstance(value, dict):
        return dict(value)
    output = str(task.get("stdout") or task.get("output") or "")
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT_JSON "):
            try:
                value = json.loads(line.split(" ", 1)[1])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def collect(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str,
) -> Path:
    plan, _root, _profile_value = _load_plan(plan_path)
    submission = gap_tuner._validate_seal(  # noqa: SLF001
        _read(submission_path), SUBMISSION_SCHEMA
    )
    if (
        submission.get("complete") is not True
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
    ):
        raise TurnGradedBatchError("submission contract drifted")
    topology_required_variants = {
        *ACTUAL_CONNECTION_VARIANT_IDS.values(),
        OPPOSED_SIDE_SENSITIVITY_VARIANT_ID,
    }
    selected_variant_ids = set(plan.get("selected_variant_ids") or [])
    topology_evidence = (
        _actual_connection_topology_evidence(plan)
        if topology_required_variants.issubset(selected_variant_ids)
        else {
            "sealed": False,
            "reason": (
                "bounded subset does not contain both actual-connection "
                "proxies and the opposed-side sensitivity control"
            ),
            "selected_variant_ids": sorted(selected_variant_ids),
            "required_variant_ids": sorted(topology_required_variants),
            "full_model_series_interconnect_attested": False,
        }
    )
    lanes = {lane["lane_index"]: lane for lane in plan["lanes"]}
    rows = []
    terminal = {"completed", "failed", "cancelled", "timeout"}
    for submitted in submission["submissions"]:
        lane = lanes[submitted["lane_index"]]
        task = _api_task(scheduler_url, int(submitted["task_id"]))
        result = _result(task)
        variant_id = lane["variant"]["id"]
        actual_connection_variant = (
            ACTUAL_CONNECTION_VARIANT_IDS.get(
                lane["variant"]["active"]
            )
            == variant_id
        )
        opposed_side_sensitivity = (
            variant_id == OPPOSED_SIDE_SENSITIVITY_VARIANT_ID
        )
        row: dict[str, Any] = {
            "lane_index": lane["lane_index"],
            "variant": lane["variant"],
            "task_id": int(submitted["task_id"]),
            "status": str(task.get("status") or "").lower(),
            "node": task.get("actual_node_name") or task.get("node_name"),
            "result_available": result is not None,
            "contract_valid": False,
            "resonance_spec_pass_15kHz": False,
            "actual_connection_design_applicable": (
                actual_connection_variant
            ),
            "sensitivity_only": not actual_connection_variant,
            "connection_topology_role": (
                "actual-current-solver-connection-proxy"
                if actual_connection_variant
                else "opposed-polarity-sensitivity-only-not-actual-design"
                if opposed_side_sensitivity
                else "connection-or-voltage-convention-sensitivity-only"
            ),
        }
        if result is not None:
            active = lane["variant"]["active"]
            prefix = "tx" if active == "Tx" else "rx"
            capacitance = result.get(
                f"C_{prefix}_{prefix}_turn_graded_F"
            )
            resonance = result.get(f"f_res_{prefix}_turn_graded_Hz")
            ltx_uH = result.get("Ltx")
            try:
                row.update(
                    {
                        "C_terminal_F": float(capacitance),
                        "f_res_Hz": float(resonance),
                        "Lm_primary_referred_H": 2.0
                        * float(ltx_uH)
                        * 1e-6,
                        "even_potential_symmetry_assumed": int(
                            result[
                                "electrostatic_even_potential_symmetry_assumed"
                            ]
                        ),
                    }
                )
                row["contract_valid"] = bool(
                    result.get("git_hash") == plan["solver_revision"]
                    and result.get("full_model") == 0
                    and result.get("round_corner") == 0
                    and result.get("cap_turn_graded_active_winding") == active
                    and float(result.get("core_center_gap_mm"))
                    == float(plan["source"]["core_center_gap_mm"])
                    and float(capacitance) > 0.0
                    and float(resonance) > 0.0
                    and row["even_potential_symmetry_assumed"] == 1
                )
                row["resonance_spec_pass_15kHz"] = bool(
                    row["contract_valid"]
                    and float(resonance) >= RESONANCE_MIN_HZ
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                row["contract_valid"] = False
        rows.append(row)
    all_terminal = all(row["status"] in terminal for row in rows)
    by_variant = {
        row["variant"]["id"]: row
        for row in rows
    }
    actual_connection_rows = {
        winding: by_variant.get(variant_id)
        for winding, variant_id in ACTUAL_CONNECTION_VARIANT_IDS.items()
    }
    actual_connection_spec_pass = all(
        row is not None and row["resonance_spec_pass_15kHz"]
        for row in actual_connection_rows.values()
    )
    actual_connection_results = {
        winding: {
            "variant_id": variant_id,
            "task_id": row["task_id"] if row is not None else None,
            "C_terminal_F": (
                row.get("C_terminal_F") if row is not None else None
            ),
            "f_res_Hz": (
                row.get("f_res_Hz") if row is not None else None
            ),
            "Lm_primary_referred_H": (
                row.get("Lm_primary_referred_H")
                if row is not None
                else None
            ),
            "contract_valid": (
                row["contract_valid"] if row is not None else False
            ),
            "resonance_spec_pass_15kHz": (
                row["resonance_spec_pass_15kHz"]
                if row is not None
                else False
            ),
        }
        for winding, variant_id in ACTUAL_CONNECTION_VARIANT_IDS.items()
        for row in (actual_connection_rows[winding],)
    }
    actual_frequencies = [
        float(row["f_res_Hz"])
        for row in actual_connection_rows.values()
        if row is not None and row.get("f_res_Hz") is not None
    ]
    opposed_row = by_variant.get(OPPOSED_SIDE_SENSITIVITY_VARIANT_ID)
    return _write(
        output,
        _seal(
            {
                "schema_version": COLLECTION_SCHEMA,
                "plan_payload_sha256": plan["payload_sha256"],
                "submission_payload_sha256": submission["payload_sha256"],
                "rows": rows,
                "all_terminal": all_terminal,
                "valid_result_count": sum(
                    bool(row["contract_valid"]) for row in rows
                ),
                "resonance_spec_pass_count_all_variants": sum(
                    bool(row["resonance_spec_pass_15kHz"])
                    for row in rows
                ),
                "resonance_spec_fail_variant_ids": [
                    row["variant"]["id"]
                    for row in rows
                    if row["contract_valid"]
                    and not row["resonance_spec_pass_15kHz"]
                ],
                "actual_connection_topology_basis": topology_evidence,
                "actual_connection_results": actual_connection_results,
                "actual_connection_fmin_Hz": (
                    min(actual_frequencies)
                    if len(actual_frequencies)
                    == len(ACTUAL_CONNECTION_VARIANT_IDS)
                    else None
                ),
                "actual_connection_resonance_spec_pass": (
                    actual_connection_spec_pass
                ),
                "opposed_side_sensitivity_result": {
                    "variant_id": OPPOSED_SIDE_SENSITIVITY_VARIANT_ID,
                    "task_id": (
                        opposed_row["task_id"]
                        if opposed_row is not None
                        else None
                    ),
                    "C_terminal_F": (
                        opposed_row.get("C_terminal_F")
                        if opposed_row is not None
                        else None
                    ),
                    "f_res_Hz": (
                        opposed_row.get("f_res_Hz")
                        if opposed_row is not None
                        else None
                    ),
                    "resonance_spec_pass_15kHz": (
                        opposed_row["resonance_spec_pass_15kHz"]
                        if opposed_row is not None
                        else False
                    ),
                    "classification": (
                        "sensitivity-only-not-actual-design"
                    ),
                },
                "scheduler_mutation_performed": False,
                "full_model_series_interconnect_attested": False,
                "final_design_pass_allowed": False,
            }
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    prepare_cmd.add_argument("--round", type=Path, default=DEFAULT_ROUND)
    prepare_cmd.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_cmd.add_argument("--solver-revision", required=True)
    prepare_cmd.add_argument("--library-revision", required=True)
    prepare_cmd.add_argument(
        "--variant-ids",
        nargs="+",
        choices=[variant["id"] for variant in VARIANTS],
        help="Optional non-empty ordered subset; defaults to the full sweep.",
    )
    submit_cmd = commands.add_parser("submit")
    submit_cmd.add_argument("--plan", type=Path, required=True)
    submit_cmd.add_argument("--output", type=Path, required=True)
    submit_cmd.add_argument("--scheduler-url", default=SCHEDULER_URL)
    collect_cmd = commands.add_parser("collect")
    collect_cmd.add_argument("--plan", type=Path, required=True)
    collect_cmd.add_argument("--submission", type=Path, required=True)
    collect_cmd.add_argument("--output", type=Path, required=True)
    collect_cmd.add_argument("--scheduler-url", default=SCHEDULER_URL)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        path = prepare(
            campaign_path=args.campaign,
            round_path=args.round,
            output=args.output,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            variant_ids=(
                tuple(args.variant_ids) if args.variant_ids is not None else None
            ),
        )
    elif args.command == "submit":
        path = submit(
            plan_path=args.plan,
            output=args.output,
            scheduler_url=args.scheduler_url,
        )
    else:
        path = collect(
            plan_path=args.plan,
            submission_path=args.submission,
            output=args.output,
            scheduler_url=args.scheduler_url,
        )
    print(json.dumps({"status": "ok", "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
