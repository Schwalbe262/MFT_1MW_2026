"""Recurring planner that observes new data without serializing worker lanes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from .orchestrator import (
    DEFAULT_MODEL_THREADS,
    PipelineOrchestrator,
    descriptor_from_active_registry,
)
from .policy import FIRST_TUNING_ROWS, MIN_MODEL_ACTIVATION_ROWS


STATUS_SCHEMA_VERSION = 1
ACTIVE_MANIFEST_SCHEMA_VERSION = 1


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value, path: str | os.PathLike[str]) -> None:
    target = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp",
        dir=os.path.dirname(target),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ContinuousController:
    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        *,
        dataset: str | os.PathLike[str],
        registry: str | os.PathLike[str],
        solver_revision: str,
        library_revision: str,
        optuna_trials: int = 200,
        model_threads: int = DEFAULT_MODEL_THREADS,
        verification_commands=None,
        status_path: str | os.PathLike[str] | None = None,
        active_manifest_path: str | os.PathLike[str] | None = None,
    ):
        self.orchestrator = orchestrator
        self.dataset = os.path.abspath(os.fspath(dataset))
        self.registry = os.path.abspath(os.fspath(registry))
        self.solver_revision = solver_revision
        self.library_revision = library_revision
        self.optuna_trials = int(optuna_trials)
        if isinstance(model_threads, bool) or int(model_threads) < 1:
            raise ValueError("model_threads must be a positive integer")
        self.model_threads = int(model_threads)
        self.verification_commands = verification_commands
        pipeline_root = self.orchestrator.store.root.parent
        self.status_path = os.path.abspath(os.fspath(
            status_path or pipeline_root / "surrogate_status.json"
        ))
        self.active_manifest_path = os.path.abspath(os.fspath(
            active_manifest_path or pipeline_root / "active_surrogate.json"
        ))
        self._cycle = 0
        self._last_raw_rows = None
        self._last_strict_rows = None
        self._last_result = None
        self._last_error = None
        self._interval_seconds = 600

    def _dataset_status(self):
        try:
            stat = os.stat(self.dataset)
        except OSError as exc:
            return {
                "path": self.dataset,
                "available": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
        return {
            "path": self.dataset,
            "available": True,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _publish_active_manifest(self, active, strict_rows: int):
        observed_at = _utc_now()
        pointer_path = os.path.join(self.registry, "current.json")
        if active is None:
            manifest = {
                "schema_version": ACTIVE_MANIFEST_SCHEMA_VERSION,
                "state": "awaiting_activation",
                "observed_at": observed_at,
                "registry": self.registry,
                "strict_full_rows": int(strict_rows),
                "activation_minimum_strict_full_rows": (
                    MIN_MODEL_ACTIVATION_ROWS
                ),
                "rows_until_activation": max(
                    0, MIN_MODEL_ACTIVATION_ROWS - int(strict_rows)
                ),
                "solver_revision": self.solver_revision,
                "library_revision": self.library_revision,
                "consumer_contract": (
                    "no production model is active; consumers must fail closed"
                ),
            }
        else:
            if not os.path.isfile(pointer_path):
                raise RuntimeError("active descriptor has no registry pointer")
            with open(pointer_path, encoding="utf-8") as handle:
                pointer = json.load(handle)
            if pointer.get("training_run_id") != active.get("training_run_id"):
                raise RuntimeError(
                    "active descriptor changed while publishing its manifest"
                )
            quality_status = os.path.abspath(
                os.fspath(active["quality_status"])
            )
            manifest = {
                "schema_version": ACTIVE_MANIFEST_SCHEMA_VERSION,
                "state": "active",
                "observed_at": observed_at,
                "model_identity": f"model:{active['training_run_id']}",
                "training_run_id": active["training_run_id"],
                "strict_full_rows": int(active["strict_full_rows"]),
                "solver_revision": self.solver_revision,
                "library_revision": self.library_revision,
                "registry": self.registry,
                "registry_pointer": pointer_path,
                "registry_pointer_sha256": _sha256(pointer_path),
                "generation": os.path.abspath(
                    os.fspath(active["generation"])
                ),
                "dataset": os.path.abspath(os.fspath(active["dataset"])),
                "dataset_sha256": pointer.get("dataset_sha256"),
                "quality_status": quality_status,
                "quality_status_sha256": pointer.get(
                    "quality_gate_sha256"
                ),
                "consumer_contract": (
                    "read this manifest once at run start, verify the pointer "
                    "and quality fingerprints, then pin the immutable "
                    "generation for the entire run"
                ),
                "next_model_contract": (
                    "a newly promoted training_run_id creates a separate "
                    "idempotent NSGA-II generation; running searches never "
                    "hot-swap models"
                ),
            }
        _atomic_json(manifest, self.active_manifest_path)
        return manifest

    def _publish_status(self, state: str, *, active_manifest=None):
        try:
            queue_stats = self.orchestrator.queue.stats()
        except Exception as exc:
            queue_stats = {"error": f"{type(exc).__name__}:{exc}"}
        value = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "pid": os.getpid(),
            "state": state,
            "cycle": int(self._cycle),
            "solver_revision": self.solver_revision,
            "library_revision": self.library_revision,
            "dataset": self._dataset_status(),
            "raw_rows": self._last_raw_rows,
            "strict_full_rows": self._last_strict_rows,
            "activation_minimum_strict_full_rows": MIN_MODEL_ACTIVATION_ROWS,
            "first_tuning_strict_full_rows": FIRST_TUNING_ROWS,
            "queue": queue_stats,
            "last_jobs": dict(
                getattr(self._last_result, "jobs", {}) or {}
            ),
            "dataset_generation": (
                getattr(self._last_result, "dataset_generation", None)
            ),
            "blocked": dict(
                getattr(self._last_result, "blocked", {}) or {}
            ),
            "last_error": self._last_error,
            "next_check_seconds": int(self._interval_seconds),
            "active_manifest": self.active_manifest_path,
            "active_model_state": (
                active_manifest.get("state") if active_manifest else None
            ),
        }
        _atomic_json(value, self.status_path)
        return value

    def inspect_strict_rows(self, dataset=None) -> int:
        training_root = str(
            Path(self.orchestrator.runtime_root) / "training"
        )
        regression_root = str(self.orchestrator.runtime_root)
        for value in (training_root, regression_root):
            if value not in sys.path:
                sys.path.insert(0, value)
        from checkpoint_orchestrator import inspect_dataset

        raw, _, strict, _ = inspect_dataset(
            os.path.abspath(os.fspath(dataset or self.dataset)),
            expected_solver_revision=self.solver_revision,
            expected_library_revision=self.library_revision,
        )
        self._last_raw_rows = int(len(raw))
        return int(len(strict))

    def _snapshot_dataset(self) -> str:
        """Copy the live parquet once; every later decision reads these bytes."""
        staging_root = self.orchestrator.store.root.parent / "dataset_staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        descriptor, staged = tempfile.mkstemp(
            prefix="dataset-", suffix=".parquet", dir=staging_root
        )
        try:
            with open(self.dataset, "rb") as source, os.fdopen(
                descriptor, "wb"
            ) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            return staged
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.remove(staged)
            except OSError:
                pass
            raise

    def plan_once(self, *, drift_detected=False, quality_regression=False):
        self._cycle += 1
        self._last_error = None
        self._publish_status("planning")
        active = None
        snapshot = None
        try:
            if os.path.isfile(os.path.join(self.registry, "current.json")):
                active = descriptor_from_active_registry(self.registry)
            snapshot = self._snapshot_dataset()
            strict_rows = self.inspect_strict_rows(snapshot)
            self._last_strict_rows = int(strict_rows)
            active_manifest = self._publish_active_manifest(
                active, strict_rows
            )
            result = self.orchestrator.plan_cycle(
                dataset_path=snapshot,
                dataset_series_path=self.dataset,
                strict_full_rows=strict_rows,
                solver_revision=self.solver_revision,
                library_revision=self.library_revision,
                active_model=active,
                drift_detected=drift_detected,
                quality_regression=quality_regression,
                optuna_trials=self.optuna_trials,
                model_threads=self.model_threads,
                verification_commands=self.verification_commands,
            )
            self._last_result = result
            self._publish_status(
                "waiting_for_next_dataset_check",
                active_manifest=active_manifest,
            )
            return result
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}:{exc}"
            self._publish_status("error")
            raise
        finally:
            if snapshot is not None:
                try:
                    os.remove(snapshot)
                except FileNotFoundError:
                    pass

    def run_forever(self, interval_seconds=600, stop_event=None):
        interval = max(30, int(interval_seconds))
        self._interval_seconds = interval
        while stop_event is None or not stop_event.is_set():
            try:
                result = self.plan_once()
                print(json.dumps(result.__dict__, sort_keys=True), flush=True)
            except Exception as exc:
                print(
                    json.dumps({
                        "pipeline_controller_error": f"{type(exc).__name__}:{exc}"
                    }),
                    file=sys.stderr,
                    flush=True,
                )
            if stop_event is not None:
                stop_event.wait(interval)
            else:
                time.sleep(interval)
