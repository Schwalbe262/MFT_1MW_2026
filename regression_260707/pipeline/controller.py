"""Recurring planner that observes new data without serializing worker lanes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from .orchestrator import PipelineOrchestrator, descriptor_from_active_registry


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
        verification_commands=None,
    ):
        self.orchestrator = orchestrator
        self.dataset = os.path.abspath(os.fspath(dataset))
        self.registry = os.path.abspath(os.fspath(registry))
        self.solver_revision = solver_revision
        self.library_revision = library_revision
        self.optuna_trials = int(optuna_trials)
        self.verification_commands = verification_commands

    def inspect_strict_rows(self, dataset=None) -> int:
        training_root = str(
            Path(self.orchestrator.runtime_root) / "training"
        )
        regression_root = str(self.orchestrator.runtime_root)
        for value in (training_root, regression_root):
            if value not in sys.path:
                sys.path.insert(0, value)
        from checkpoint_orchestrator import inspect_dataset

        _, _, strict, _ = inspect_dataset(
            os.path.abspath(os.fspath(dataset or self.dataset)),
            expected_solver_revision=self.solver_revision,
            expected_library_revision=self.library_revision,
        )
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
        active = None
        if os.path.isfile(os.path.join(self.registry, "current.json")):
            active = descriptor_from_active_registry(self.registry)
        snapshot = self._snapshot_dataset()
        try:
            strict_rows = self.inspect_strict_rows(snapshot)
            return self.orchestrator.plan_cycle(
                dataset_path=snapshot,
                dataset_series_path=self.dataset,
                strict_full_rows=strict_rows,
                solver_revision=self.solver_revision,
                library_revision=self.library_revision,
                active_model=active,
                drift_detected=drift_detected,
                quality_regression=quality_regression,
                optuna_trials=self.optuna_trials,
                verification_commands=self.verification_commands,
            )
        finally:
            try:
                os.remove(snapshot)
            except FileNotFoundError:
                pass

    def run_forever(self, interval_seconds=600, stop_event=None):
        interval = max(30, int(interval_seconds))
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
