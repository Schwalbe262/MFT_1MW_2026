"""Generation planner for overlapping MFT pipeline stages.

The planner is side-effect free with respect to Slurm/AEDT.  It snapshots the
current dataset, then idempotently records commands in the durable queue.
Workers in :mod:`pipeline.runner` execute those commands later.  Running the
planner repeatedly is therefore the normal control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

from .artifacts import GenerationStore
from .policy import (
    NSGA_MAX_WORKERS,
    NSGA_POPULATION,
    NSGA_RESTARTS,
    FINE_VERIFICATION_COUNT,
    STANDARD_VERIFICATION_COUNT,
    next_training_checkpoint,
    tuning_decision,
)
from .queue import DurableJobQueue, Job


DEFAULT_MODEL_THREADS = 24


@dataclass(frozen=True)
class CycleResult:
    dataset_generation: str
    jobs: dict[str, int]
    blocked: dict[str, str] = field(default_factory=dict)


class PipelineOrchestrator:
    """Translate immutable input generations into an idempotent job DAG."""

    def __init__(
        self,
        queue: DurableJobQueue,
        store: GenerationStore,
        runtime_root: str | os.PathLike[str],
        *,
        python: str = sys.executable,
    ):
        self.queue = queue
        self.store = store
        self.runtime_root = Path(runtime_root).resolve()
        self.python = os.path.abspath(python)

    def latest_tuning(
        self,
        solver_revision: str,
        library_revision: str,
        data_contract_sha256: str,
    ) -> tuple[int, str, str] | None:
        candidates = []
        for generation in self.store.generations("tuning"):
            metadata = generation.manifest.get("metadata", {})
            rows = metadata.get("strict_full_rows")
            params = generation.path / "params.json"
            if (
                isinstance(rows, int)
                and params.is_file()
                and metadata.get("solver_revision") == solver_revision
                and metadata.get("library_revision") == library_revision
                and metadata.get("data_contract_sha256")
                == data_contract_sha256
            ):
                candidates.append(
                    (rows, str(generation.path), generation.generation_id)
                )
        return max(candidates, default=None, key=lambda item: (item[0], item[2]))

    def plan_cycle(
        self,
        *,
        dataset_path: str | os.PathLike[str],
        dataset_series_path: str | os.PathLike[str] | None = None,
        strict_full_rows: int,
        solver_revision: str,
        library_revision: str,
        active_model: Mapping[str, object] | None = None,
        drift_detected: bool = False,
        quality_regression: bool = False,
        collect_interval_seconds: int = 600,
        optuna_trials: int = 200,
        model_threads: int = DEFAULT_MODEL_THREADS,
        verification_commands: Mapping[str, Mapping[str, object]] | None = None,
        now: float | None = None,
    ) -> CycleResult:
        rows = int(strict_full_rows)
        if rows < 0:
            raise ValueError("strict_full_rows must be non-negative")
        if isinstance(model_threads, bool) or int(model_threads) < 1:
            raise ValueError("model_threads must be a positive integer")
        model_threads = int(model_threads)
        for label, revision in (
            ("solver", solver_revision), ("library", library_revision)
        ):
            value = str(revision or "").lower()
            if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{label} revision must be a full SHA")

        training_dir = str(Path(__file__).resolve().parents[1] / "training")
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        from checkpoint_contract import (
            checkpoint_contract_identity,
            checkpoint_status_revision_identity_matches,
        )

        contract = checkpoint_contract_identity(
            solver_revision=solver_revision.lower()
        )
        data_contract_sha256 = contract["checkpoint_contract_sha256"]
        cohort_key = (
            f"{solver_revision.lower()}:{library_revision.lower()}:"
            f"{data_contract_sha256}"
        )

        dataset = self.store.publish_files(
            "dataset",
            {"train.parquet": dataset_path},
            metadata={
                "strict_full_rows": rows,
                "solver_revision": solver_revision.lower(),
                "library_revision": library_revision.lower(),
            },
        )
        dataset_identity = f"dataset:{dataset.generation_id}"
        timestamp = time.time() if now is None else float(now)
        bucket = int(timestamp // max(1, int(collect_interval_seconds)))
        jobs: dict[str, int] = {}
        collect = self.queue.enqueue(
            "collect",
            f"collector-v2-window-{bucket}",
            {
                "command": [
                    self.python,
                    str(self.runtime_root / "campaign" / "collect_wave.py"),
                    "--prefix", "mft-camp",
                    "--extra-prefix", "mft-1to3",
                    "--extra-prefix", "mft-1x3",
                    "--extra-prefix", "mft-mixed",
                    "--extra-prefix", "mft-9way",
                    "--running-fetch-limit", "0",
                ],
                "cwd": str(self.runtime_root),
                "retry": True,
                "retry_backoff_seconds": 60,
            },
            priority=100,
            max_attempts=5,
            now=timestamp,
        )
        jobs["collect"] = collect.id

        latest_tuning = self.latest_tuning(
            solver_revision.lower(),
            library_revision.lower(),
            data_contract_sha256,
        )
        tune = tuning_decision(
            rows,
            last_tuned_rows=(latest_tuning[0] if latest_tuning else None),
            drift_detected=drift_detected,
            quality_regression=quality_regression,
        )
        tune_job: Job | None = None
        params_path = latest_tuning[1] + os.sep + "params.json" if latest_tuning else None
        if tune.due:
            result_json = "{work_dir}" + os.sep + "tuning_result.json"
            tune_job = self.queue.enqueue(
                "tune",
                f"tune-{dataset.generation_id}",
                {
                    "command": [
                        self.python,
                        str(self.runtime_root / "training" / "tune_optuna.py"),
                        "--all",
                        "--trials", str(int(optuna_trials)),
                        "--model-threads", str(model_threads),
                        "--dataset", str(dataset.path / "train.parquet"),
                        "--artifact-root", str(self.store.root),
                        "--result-json", result_json,
                        "--solver-revision", solver_revision.lower(),
                        "--library-revision", library_revision.lower(),
                        "--data-contract-sha256", data_contract_sha256,
                    ],
                    "cwd": str(self.runtime_root),
                    "result_json": result_json,
                    "result_output_key": "generation_path",
                    "result_generation_kind": "tuning",
                    "result_generation_id_key": "generation_id",
                    "retry": True,
                    "retry_backoff_seconds": 300,
                },
                input_generation=dataset_identity,
                coalesce_key=cohort_key,
                coalesce_pending=True,
                priority=70,
                max_attempts=3,
                now=timestamp,
            )
            jobs["tune"] = tune_job.id
        else:
            self.queue.cancel_coalesced_pending(
                "tune", cohort_key, "tuning_no_longer_due"
            )

        checkpoint_run_root = (
            self.runtime_root
            / "training"
            / "checkpoint_runs"
            / (
                f"{library_revision.lower()}-c"
                f"{contract['checkpoint_contract_key']}"
            )
        )
        active_rows = int((active_model or {}).get("strict_full_rows") or 0)
        status_path = self.runtime_root / "training" / "strict_data_status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                identity = status.get("state_identity") or {}
                if (
                    checkpoint_status_revision_identity_matches(
                        status, solver_revision, library_revision
                    )
                    and identity.get("checkpoint_contract_key")
                    == contract["checkpoint_contract_key"]
                ):
                    active_rows = max(
                        active_rows,
                        max(
                            [int(value) for value in status.get(
                                "completed_thresholds", []
                            )],
                            default=0,
                        ),
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        # ``strict_data_status.json`` is written before a checkpoint starts, so
        # it can legitimately lag after a successful long-running worker.  The
        # identity-scoped checkpoint state is the durable completion ledger;
        # read it on every controller cycle so 500 -> 1000 -> 2000 progresses
        # without waiting for the dataset bytes to change.
        checkpoint_state_path = checkpoint_run_root / "checkpoint_state.json"
        if checkpoint_state_path.is_file():
            try:
                checkpoint_state = json.loads(
                    checkpoint_state_path.read_text(encoding="utf-8")
                )
                identity = checkpoint_state.get("identity") or {}
                identity_matches = (
                    checkpoint_state.get("schema_version") == 2
                    and identity.get("checkpoint_contract_sha256")
                    == contract["checkpoint_contract_sha256"]
                    and identity.get("checkpoint_contract_key")
                    == contract["checkpoint_contract_key"]
                    and identity.get("solver_revision_cohort")
                    == contract["solver_revision_cohort"]
                    and identity.get("physics_data_revision")
                    == contract["physics_data_revision"]
                    and str(identity.get("library_revision") or "").lower()
                    == library_revision.lower()
                )
                if identity_matches:
                    completed_thresholds = [
                        int(item["threshold"])
                        for item in checkpoint_state.get("completed", [])
                        if isinstance(item, dict)
                        and isinstance(item.get("threshold"), int)
                        and not isinstance(item.get("threshold"), bool)
                        and 0 < int(item["threshold"]) <= rows
                    ]
                    active_rows = max(
                        active_rows,
                        max(completed_thresholds, default=0),
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        checkpoint = next_training_checkpoint(rows, active_rows)
        if checkpoint is not None:
            dependencies = [tune_job.id] if tune_job else []
            params_argument: list[str] = []
            if tune_job:
                params_argument = [
                    "--params",
                    "{dependency_tune_output}" + os.sep + "params.json",
                ]
            elif params_path:
                params_argument = ["--params", params_path]
            train_command = [
                self.python,
                str(self.runtime_root / "training" / "checkpoint_orchestrator.py"),
                "--runtime-root", str(self.runtime_root),
                "--dataset", str(dataset.path / "train.parquet"),
                "--dataset-series", os.path.abspath(
                    os.fspath(dataset_series_path or dataset_path)
                ),
                "--output-root", str(self.runtime_root / "training"),
                "--run-root", str(checkpoint_run_root),
                "--execute",
                "--solver-revision", solver_revision.lower(),
                "--library-revision", library_revision.lower(),
                "--source-dataset-generation", dataset_identity,
            ] + params_argument
            train = self.queue.enqueue(
                "train",
                f"checkpoint-{checkpoint}-{dataset.generation_id}",
                {
                    "command": train_command,
                    "cwd": str(self.runtime_root),
                    "env": {
                        "MFT_PIPELINE_ROOT": str(self.store.root.parent),
                    },
                    "dependency_kinds": ({"tune": "tuning"} if tune_job else {}),
                    "retry": True,
                    "retry_backoff_seconds": 600,
                },
                input_generation=dataset_identity,
                coalesce_key=cohort_key,
                coalesce_pending=True,
                dependencies=dependencies,
                priority=80,
                max_attempts=3,
                now=timestamp,
            )
            jobs["train"] = train.id
        else:
            self.queue.cancel_coalesced_pending(
                "train", cohort_key, "checkpoint_no_longer_due"
            )

        if active_model:
            required = (
                "training_run_id", "generation", "dataset", "quality_status"
            )
            missing = [key for key in required if not active_model.get(key)]
            if missing:
                raise ValueError(f"active model descriptor is missing: {missing}")
            model_id = str(active_model["training_run_id"])
            optimize = self.queue.enqueue(
                "optimize",
                f"nsga-{model_id}",
                {
                    "command": [
                        self.python,
                        str(self.runtime_root / "optimization" / "run_nsga2.py"),
                        "--restarts", str(NSGA_RESTARTS),
                        "--pop", str(NSGA_POPULATION),
                        "--workers", str(NSGA_MAX_WORKERS),
                        "--round", "0",
                        "--dataset", str(active_model["dataset"]),
                        "--registry", str(active_model.get(
                            "registry", self.runtime_root / "training" / "registry"
                        )),
                        "--registry-generation", str(active_model["generation"]),
                        "--quality-status", str(active_model["quality_status"]),
                        "--output-root", "{work_dir}",
                    ],
                    "cwd": str(self.runtime_root),
                    "publish": {
                        "kind": "optimization",
                        "source": "{work_dir}" + os.sep + "round_00",
                        "metadata": {
                            "training_run_id": model_id,
                            "strict_full_rows": active_rows,
                            "restarts": NSGA_RESTARTS,
                            "population": NSGA_POPULATION,
                            "workers": NSGA_MAX_WORKERS,
                        },
                        "parents": [f"model:{model_id}"],
                    },
                    "retry": True,
                    "retry_backoff_seconds": 300,
                },
                input_generation=f"model:{model_id}",
                priority=50,
                max_attempts=2,
                now=timestamp,
            )
            jobs["optimize"] = optimize.id

            commands = dict(verification_commands or {})
            unknown_commands = set(commands) - {"standard", "fine"}
            if unknown_commands:
                raise ValueError(
                    f"unknown verification command stages: {sorted(unknown_commands)}"
                )
            normalized_commands = {}
            for stage, command in commands.items():
                if (
                    not isinstance(command, Mapping)
                    or command.get("adapter") != "mft_scheduler_v1"
                    or command.get("execute") is not True
                ):
                    raise ValueError(
                        f"{stage} verification must use the explicitly enabled "
                        "mft_scheduler_v1 adapter"
                    )
                config = dict(command)
                library_root = os.path.abspath(
                    os.fspath(config.get("library_root") or "")
                )
                if not os.path.isdir(library_root):
                    raise ValueError(
                        f"{stage} verification library_root is unavailable"
                    )
                config["library_root"] = library_root
                normalized_commands[stage] = config
            commands = normalized_commands
            blocked: dict[str, str] = {}
            if "standard" in commands:
                standard = self.queue.enqueue(
                    "verify_standard",
                    f"standard-{model_id}",
                    {
                        "command": [
                            self.python,
                            "-m", "pipeline.verification_adapter",
                            "--stage", "standard",
                            "--input-generation", "{dependency_optimize_output}",
                            "--output-dir", "{work_dir}" + os.sep + "verified",
                            "--expected-count", str(STANDARD_VERIFICATION_COUNT),
                            "--adapter-config-json", json.dumps(
                                commands["standard"],
                                separators=(",", ":"),
                            ),
                        ],
                        "cwd": str(self.runtime_root),
                        "dependency_kinds": {"optimize": "optimization"},
                        "publish": {
                            "kind": "verification_standard",
                            "source": "{work_dir}" + os.sep + "verified",
                            "metadata": {
                                "training_run_id": model_id,
                                "count": STANDARD_VERIFICATION_COUNT,
                                "selection_policy": "reviewed_exact_count_v1",
                            },
                        },
                        "retry": True,
                        "retry_backoff_seconds": 300,
                    },
                    input_generation=f"model:{model_id}",
                    dependencies=[optimize.id],
                    priority=40,
                    max_attempts=2,
                    now=timestamp,
                )
                jobs["verify_standard"] = standard.id
                if "fine" in commands:
                    fine = self.queue.enqueue(
                        "verify_fine",
                        f"fine-{model_id}",
                        {
                            "command": [
                                self.python,
                                "-m", "pipeline.verification_adapter",
                                "--stage", "fine",
                                "--input-generation",
                                "{dependency_verify_standard_output}",
                                "--output-dir", "{work_dir}" + os.sep + "verified",
                                "--expected-count", str(FINE_VERIFICATION_COUNT),
                                "--adapter-config-json", json.dumps(
                                    commands["fine"],
                                    separators=(",", ":"),
                                ),
                            ],
                            "cwd": str(self.runtime_root),
                            "dependency_kinds": {
                                "verify_standard": "verification_standard"
                            },
                            "publish": {
                                "kind": "verification_fine",
                                "source": "{work_dir}" + os.sep + "verified",
                                "metadata": {
                                    "training_run_id": model_id,
                                    "count": FINE_VERIFICATION_COUNT,
                                    "selection_policy": "reviewed_exact_count_v1",
                                },
                            },
                            "retry": True,
                            "retry_backoff_seconds": 300,
                        },
                        input_generation=f"model:{model_id}",
                        dependencies=[standard.id],
                        priority=45,
                        max_attempts=2,
                        now=timestamp,
                    )
                    jobs["verify_fine"] = fine.id
                else:
                    blocked["verification_fine"] = "fine_command_not_configured"
            else:
                blocked["verification_standard"] = (
                    "standard_command_not_configured"
                )
                blocked["verification_fine"] = "standard_verification_blocked"

            return CycleResult(dataset_identity, jobs, blocked)

        return CycleResult(dataset_identity, jobs)


def descriptor_from_active_registry(registry: str | os.PathLike[str]) -> dict:
    """Return a pinned descriptor accepted by ``plan_cycle``."""
    registry = os.path.abspath(os.fspath(registry))
    training_dir = str(Path(__file__).resolve().parents[1] / "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from train_models import load_active_generation

    active = load_active_generation(registry)
    report = active["report"]
    dataset = os.path.abspath(report["dataset_path"])
    if not os.path.isfile(dataset):
        raise RuntimeError("active model's immutable dataset snapshot is missing")
    return {
        "training_run_id": report["training_run_id"],
        "strict_full_rows": int(report["strict_full_rows"]),
        "generation": active["generation"],
        "registry": registry,
        "dataset": dataset,
        "quality_status": os.path.join(active["generation"], "quality_gate.json"),
    }
