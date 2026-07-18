"""Generation planner for overlapping MFT pipeline stages.

The planner is side-effect free with respect to Slurm/AEDT.  It snapshots the
current dataset, then idempotently records commands in the durable queue.
Workers in :mod:`pipeline.runner` execute those commands later.  Running the
planner repeatedly is therefore the normal control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, Mapping

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
        checkpoint_output_root: str | os.PathLike[str] | None = None,
        solver_git_repo: str | os.PathLike[str] | None = None,
    ):
        self.queue = queue
        self.store = store
        self.runtime_root = Path(runtime_root).resolve()
        self.python = os.path.abspath(python)
        self.checkpoint_output_root = Path(
            checkpoint_output_root or self.runtime_root / "training"
        ).resolve()
        configured_solver_git_repo = str(
            solver_git_repo or os.environ.get("MFT_SOLVER_GIT_REPO") or ""
        ).strip()
        self.solver_git_repo = (
            Path(configured_solver_git_repo).resolve()
            if configured_solver_git_repo
            else None
        )

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

    def active_coalesced_job(
        self,
        job_type: str,
        coalesce_key: str,
        *,
        idempotency_prefix: str | None = None,
        idempotency_suffix: str | None = None,
        payload_match: Callable[[Mapping[str, object]], bool] | None = None,
        pending_input_generation: str | None = None,
        required_dependency_ids: tuple[int, ...] | None = None,
        now: float | None = None,
    ) -> Job | None:
        """Return one already-owned cohort job instead of replacing its input.

        Production tuning is intentionally long-running.  Replacing its queued
        successor on every collector snapshot also replaced the dependent train
        job.  A running job is the cohort authority; otherwise the newest
        matching pending job is reused.  Train callers may require the current
        dataset generation and an exact dependency inventory so an obsolete
        dependency-bound checkpoint cannot be mistaken for an independent
        checkpoint.  Re-enqueuing that exact durable row atomically removes
        stale pending siblings while preserving its immutable input and any
        active worker lease.
        """

        candidates = [
            job
            for job in self.queue.list(
                states=["running", "queued", "retry_wait"],
                job_types=[job_type],
            )
            if job.coalesce_key == coalesce_key
            and (
                idempotency_prefix is None
                or job.idempotency_key.startswith(idempotency_prefix)
            )
            and (
                idempotency_suffix is None
                or job.idempotency_key.endswith(idempotency_suffix)
            )
            and (payload_match is None or payload_match(job.payload))
        ]
        dependency_jobs = {
            job.id: self.queue.dependencies(job.id)
            for job in candidates
        }
        if required_dependency_ids is not None:
            required = tuple(sorted(int(value) for value in required_dependency_ids))
            candidates = [
                job
                for job in candidates
                if tuple(
                    sorted(
                        dependency.id
                        for dependency in dependency_jobs[job.id]
                    )
                )
                == required
            ]
        # A legacy successor may depend on a pending tune that was just
        # cancelled when the older running tune became authoritative.  Never
        # revive that dead branch.  Zero-dependency and succeeded-dependency
        # jobs remain valid across the tune-due -> tune-complete transition.
        candidates = [
            job
            for job in candidates
            if all(
                dependency.state not in {"failed", "cancelled"}
                for dependency in dependency_jobs[job.id]
            )
            and (
                not dependency_jobs[job.id]
                if job_type == "tune"
                else all(
                    dependency.job_type == "tune"
                    and dependency.coalesce_key == coalesce_key
                    for dependency in dependency_jobs[job.id]
                )
            )
        ]
        running = [job for job in candidates if job.state == "running"]
        if running:
            authority = max(running, key=lambda job: job.id)
        else:
            pending = [
                job
                for job in candidates
                if job.state in {"queued", "retry_wait"}
                and (
                    pending_input_generation is None
                    or job.input_generation == pending_input_generation
                )
            ]
            authority = max(pending, default=None, key=lambda job: job.id)
        if authority is None:
            return None

        reused = self.queue.enqueue(
            authority.job_type,
            authority.idempotency_key,
            authority.payload,
            input_generation=authority.input_generation,
            coalesce_key=authority.coalesce_key,
            coalesce_pending=True,
            dependencies=[
                dependency.id for dependency in dependency_jobs[authority.id]
            ],
            priority=authority.priority,
            max_attempts=authority.max_attempts,
            now=now,
        )
        if reused.state in {"running", "queued", "retry_wait"}:
            return reused
        return None

    @staticmethod
    def _script_identity(path: str | os.PathLike[str]) -> str:
        """Return a path-independent identity when script bytes are available."""

        script = Path(path)
        try:
            if script.is_file():
                return "sha256:" + hashlib.sha256(script.read_bytes()).hexdigest()
        except OSError:
            pass
        # Missing scripts are never considered equivalent across deployments.
        # This keeps test/developer fixtures deterministic and fails closed if
        # an immutable deployment has already been removed.
        return "missing:" + os.path.normcase(os.path.abspath(os.fspath(script)))

    @classmethod
    def _tune_contract_from_command(
        cls, command: object
    ) -> dict[str, object] | None:
        if (
            not isinstance(command, list)
            or len(command) < 2
            or not all(isinstance(value, str) for value in command)
        ):
            return None

        def integer_option(name: str) -> int:
            index = command.index(name)
            return int(command[index + 1])

        try:
            return {
                "kind": "pipeline_tune_v1",
                "python": os.path.normcase(os.path.abspath(command[0])),
                "script_identity": cls._script_identity(command[1]),
                "all_models": "--all" in command,
                "trials": integer_option("--trials"),
                "model_threads": integer_option("--model-threads"),
                "job_workers": integer_option("--job-workers"),
                "max_model_thread_budget": integer_option(
                    "--max-model-thread-budget"
                ),
            }
        except (IndexError, TypeError, ValueError):
            return None

    @classmethod
    def _tune_contract_from_payload(
        cls, payload: Mapping[str, object]
    ) -> dict[str, object] | None:
        declared = payload.get("execution_contract")
        if (
            isinstance(declared, dict)
            and declared.get("kind") == "pipeline_tune_v1"
        ):
            return dict(declared)
        return cls._tune_contract_from_command(payload.get("command"))

    @staticmethod
    def _collector_contract_from_payload(
        payload: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Return the deployment-independent identity of one full scan.

        The command's Python and script paths deliberately do not participate.
        Immutable controller generations move those paths even when the scan
        scope is identical.  The reviewed solver repository *does* participate
        because probe ancestry classification is part of the data contract.
        """

        command = payload.get("command")
        environment = payload.get("env")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or not all(isinstance(value, str) for value in command)
            or not isinstance(environment, dict)
        ):
            return None
        solver_git_repo = str(
            environment.get("MFT_SOLVER_GIT_REPO") or ""
        ).strip()
        collector_dataset_dir = str(
            environment.get("MFT_COLLECTOR_DATASET_DIR") or ""
        ).strip()
        if not solver_git_repo or not collector_dataset_dir:
            return None

        def option(name: str) -> str:
            index = command.index(name)
            return command[index + 1]

        try:
            prefixes = [option("--prefix")]
            prefixes.extend(
                command[index + 1]
                for index, value in enumerate(command[:-1])
                if value == "--extra-prefix"
            )
            running_fetch_limit = int(option("--running-fetch-limit"))
        except (IndexError, TypeError, ValueError):
            return None
        if running_fetch_limit != 0:
            return None
        return {
            "kind": "pipeline_collect_full_scan_v1",
            "prefixes": sorted(set(prefixes)),
            "running_fetch_limit": 0,
            "solver_git_repo": os.path.normcase(
                str(Path(solver_git_repo).resolve())
            ),
            "collector_dataset_dir": os.path.normcase(
                str(Path(collector_dataset_dir).resolve())
            ),
        }

    @staticmethod
    def _collector_bucket(job: Job) -> int | None:
        demand = job.payload.get("collector_demand")
        if isinstance(demand, dict):
            try:
                return int(demand["bucket"])
            except (KeyError, TypeError, ValueError):
                return None
        marker = "-window-"
        if marker not in job.idempotency_key:
            return None
        try:
            return int(job.idempotency_key.rsplit(marker, 1)[1])
        except ValueError:
            return None

    def active_semantic_collector(
        self,
        execution_contract: Mapping[str, object],
        *,
        dataset_identity: str,
        bucket: int,
    ) -> Job | None:
        """Reuse the current scan demand across immutable deployments.

        A scan can outlive several collection windows.  Repeated controller
        cycles for the same dataset/window reuse the existing authority.  A
        newer window or dataset generation is allowed to create one pending
        follow-up; enqueue coalescing replaces that pending row as still newer
        demand arrives without disturbing the running scan.
        """

        candidates = [
            job
            for job in self.queue.list(
                states=["running", "queued", "retry_wait", "succeeded"],
                job_types=["collect"],
            )
            if self._collector_contract_from_payload(job.payload)
            == dict(execution_contract)
        ]
        exact = [
            job
            for job in candidates
            if self._collector_bucket(job) == int(bucket)
            and (
                job.input_generation == dataset_identity
                # v4 jobs predate dataset-bound collection demand.  Treat a
                # running legacy scan as the authority during the v5 rollout;
                # its fixed canonical-repository contract still matches.
                or (job.input_generation is None and job.state == "running")
            )
        ]
        active_exact = [
            job
            for job in exact
            if job.state in {"running", "queued", "retry_wait"}
        ]
        if active_exact:
            return max(
                active_exact,
                key=lambda job: (job.state == "running", job.id),
            )
        succeeded_exact = [job for job in exact if job.state == "succeeded"]
        if succeeded_exact:
            return max(succeeded_exact, key=lambda job: job.id)
        return None

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
        runtime_execution_key = hashlib.sha256(json.dumps(
            {
                "python": os.path.normcase(self.python),
                "runtime_root": os.path.normcase(str(self.runtime_root)),
            },
            sort_keys=True,
        ).encode("utf-8")).hexdigest()[:12]
        jobs: dict[str, int] = {}
        collector_payload = {
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
            "env": (
                {
                    "MFT_SOLVER_GIT_REPO": str(self.solver_git_repo),
                    "MFT_COLLECTOR_DATASET_DIR": str(
                        Path(dataset_series_path or dataset_path).resolve().parent
                    ),
                }
                if self.solver_git_repo is not None
                else {}
            ),
            "collector_demand": {
                "bucket": bucket,
                "dataset_generation": dataset_identity,
            },
            "retry": True,
            "retry_backoff_seconds": 60,
        }
        collector_contract = self._collector_contract_from_payload(
            collector_payload
        )
        collect = (
            self.active_semantic_collector(
                collector_contract,
                dataset_identity=dataset_identity,
                bucket=bucket,
            )
            if collector_contract is not None
            else None
        )
        if collect is None:
            collector_contract_key = hashlib.sha256(json.dumps(
                collector_contract or {
                    "runtime_execution_key": runtime_execution_key,
                },
                sort_keys=True,
            ).encode("utf-8")).hexdigest()[:12]
            collector_payload["execution_contract"] = collector_contract
            collect = self.queue.enqueue(
                "collect",
                (
                    f"collector-v4-e{runtime_execution_key}"
                    f"-c{collector_contract_key}"
                    f"-d{dataset.generation_id[:12]}-window-{bucket}"
                ),
                collector_payload,
                input_generation=dataset_identity,
                coalesce_key=f"collector-full-scan:{collector_contract_key}",
                coalesce_pending=True,
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
            tune_script = self.runtime_root / "training" / "tune_optuna.py"
            tune_execution_contract = self._tune_contract_from_command([
                self.python,
                str(tune_script),
                "--all",
                "--trials", str(int(optuna_trials)),
                "--model-threads", "1",
                "--job-workers", str(model_threads),
                "--max-model-thread-budget", str(model_threads),
            ])
            if tune_execution_contract is None:
                raise RuntimeError("invalid tune execution contract")
            tune_execution_key = hashlib.sha256(json.dumps(
                tune_execution_contract,
                sort_keys=True,
            ).encode("utf-8")).hexdigest()[:12]
            tune_job = self.active_coalesced_job(
                "tune",
                cohort_key,
                payload_match=lambda payload: (
                    self._tune_contract_from_payload(payload)
                    == tune_execution_contract
                ),
                now=timestamp,
            )
            if tune_job is None:
                result_json = "{work_dir}" + os.sep + "tuning_result.json"
                tune_job = self.queue.enqueue(
                    "tune",
                    f"tune-{dataset.generation_id}-e{tune_execution_key}",
                    {
                        "command": [
                            self.python,
                            str(tune_script),
                            "--all",
                            "--trials", str(int(optuna_trials)),
                            "--model-threads", "1",
                            "--job-workers", str(model_threads),
                            "--max-model-thread-budget", str(model_threads),
                            "--dataset", str(dataset.path / "train.parquet"),
                            "--artifact-root", str(self.store.root),
                            "--result-json", result_json,
                            "--solver-revision", solver_revision.lower(),
                            "--library-revision", library_revision.lower(),
                            "--data-contract-sha256", data_contract_sha256,
                        ],
                        "cwd": str(self.runtime_root),
                        "execution_contract": tune_execution_contract,
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
        status_path = self.checkpoint_output_root / "strict_data_status.json"
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
            checkpoint_execution_key = hashlib.sha256(json.dumps(
                {
                    "model_threads": model_threads,
                    "output_root": os.path.normcase(
                        str(self.checkpoint_output_root)
                    ),
                    "runtime_execution_key": runtime_execution_key,
                    "script_sha256": (
                        hashlib.sha256(
                            (
                                self.runtime_root
                                / "training"
                                / "checkpoint_orchestrator.py"
                            ).read_bytes()
                        ).hexdigest()
                        if (
                            self.runtime_root
                            / "training"
                            / "checkpoint_orchestrator.py"
                        ).is_file()
                        else "missing"
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")).hexdigest()[:12]
            train = self.active_coalesced_job(
                "train",
                cohort_key,
                idempotency_prefix=f"checkpoint-{checkpoint}-",
                idempotency_suffix=f"-e{checkpoint_execution_key}",
                pending_input_generation=dataset_identity,
                required_dependency_ids=(),
                now=timestamp,
            )
            if train is None:
                params_argument: list[str] = []
                # Full-cohort HPO runs in its own durable lane and can take
                # many hours.  The newest checkpoint must therefore train in
                # parallel using the latest *completed* authenticated tuning
                # generation, or reviewed defaults for the first wave.  A
                # later checkpoint consumes newly published parameters.  The
                # checkpoint quality/promotion gates remain unchanged and
                # continue to fail closed.
                if params_path:
                    params_argument = ["--params", params_path]
                train_command = [
                    self.python,
                    str(
                        self.runtime_root
                        / "training"
                        / "checkpoint_orchestrator.py"
                    ),
                    "--runtime-root", str(self.runtime_root),
                    "--dataset", str(dataset.path / "train.parquet"),
                    "--dataset-series", os.path.abspath(
                        os.fspath(dataset_series_path or dataset_path)
                    ),
                    "--output-root", str(self.checkpoint_output_root),
                    "--run-root", str(checkpoint_run_root),
                    "--execute",
                    "--solver-revision", solver_revision.lower(),
                    "--library-revision", library_revision.lower(),
                    "--source-dataset-generation", dataset_identity,
                    "--model-threads", str(model_threads),
                ] + params_argument
                train = self.queue.enqueue(
                    "train",
                    (
                        f"checkpoint-{checkpoint}-{dataset.generation_id}"
                        "-t0"
                        f"-e{checkpoint_execution_key}"
                    ),
                    {
                        "command": train_command,
                        "cwd": str(self.runtime_root),
                        "env": {
                            "MFT_PIPELINE_ROOT": str(self.store.root.parent),
                        },
                        "dependency_kinds": {},
                        "retry": True,
                        "retry_backoff_seconds": 600,
                    },
                    input_generation=dataset_identity,
                    coalesce_key=cohort_key,
                    coalesce_pending=True,
                    dependencies=[],
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


def descriptor_from_active_registry(
    registry: str | os.PathLike[str],
    *,
    solver_revision: str | None = None,
    library_revision: str | None = None,
) -> dict:
    """Return a pinned descriptor accepted by ``plan_cycle``."""
    registry = os.path.abspath(os.fspath(registry))
    training_dir = str(Path(__file__).resolve().parents[1] / "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from train_models import load_active_generation

    active = load_active_generation(registry)
    report = active["report"]
    quality = active["quality"]
    for label, expected in (
        ("solver", solver_revision),
        ("library", library_revision),
    ):
        if expected is None:
            continue
        actual = quality.get(f"{label}_revision")
        if actual != str(expected).lower():
            raise RuntimeError(
                f"active model {label} revision does not match controller"
            )
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
