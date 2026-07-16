"""Persistent, debounced experimental HPO -> surrogate refresh controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from filelock import FileLock, Timeout
import psutil

try:
    from .experimental_surrogate import atomic_json, sha256_file
except ImportError:
    from experimental_surrogate import atomic_json, sha256_file


TARGETS = (
    "Llt_phys",
    "P_Rx_main_group",
    "B_max_core",
    "Tprobe_core_center_leg_max",
)
TERMINAL_PHASES = {
    "candidate_promoted",
    "candidate_rejected",
}


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except (OSError, ValueError, TypeError):
        return default


def _alive(pid):
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (TypeError, ValueError, psutil.Error):
        return False


def refresh_due(strict_rows, last_attempted_rows, minimum_new_rows):
    strict_rows = int(strict_rows)
    minimum_new_rows = max(1, int(minimum_new_rows))
    if strict_rows < 2000:
        return False, 2000
    if last_attempted_rows is None:
        return True, 2000
    threshold = int(last_attempted_rows) + minimum_new_rows
    return strict_rows >= threshold, threshold


class ContinuousExperimentalRefresh:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.runtime_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.state_path = self.root / "state.json"
        self.stop_path = self.root / "STOP"
        self.state = _read_json(self.state_path, {}) or {}
        self.state.setdefault("schema_version", 1)
        self.state.setdefault("last_attempted_strict_rows", None)
        self.state.setdefault("last_attempted_generation", None)
        self.state.setdefault("failure_count", 0)
        self._last_observation = None

    def _publish(self, state, **extra):
        active_status = None
        active_path = self.state.get("active_wave_status")
        if active_path:
            active_status = _read_json(active_path)
        value = {
            "schema_version": 1,
            "updated_at": _now(),
            "pid": os.getpid(),
            "state": state,
            "lane": "experimental",
            "eligibility": "FEA-NOT-APPROVED",
            "canonical_dataset": os.path.abspath(self.args.dataset),
            "solver_revision": self.args.solver_revision,
            "library_revision": self.args.library_revision,
            "minimum_new_strict_rows": self.args.min_new_rows,
            "last_attempted_strict_rows": self.state.get(
                "last_attempted_strict_rows"
            ),
            "last_attempted_generation": self.state.get(
                "last_attempted_generation"
            ),
            "active_wave": self.state.get("active_wave"),
            "active_wave_status": active_path,
            "active_wave_detail": active_status,
            "hpo_contract": {
                "targets": list(TARGETS),
                "trials_per_target": self.args.trials,
                "hpo_threads_per_target": self.args.hpo_threads,
                "total_hpo_thread_budget": len(TARGETS) * self.args.hpo_threads,
                "candidate_target_workers": self.args.target_workers,
                "candidate_model_threads": self.args.model_threads,
                "candidate_total_thread_budget": (
                    self.args.target_workers * self.args.model_threads
                ),
                "replacement": (
                    "candidate aggregate loss must improve >=0.5%; no target "
                    "may regress >5%; running NSGA searches never hot-swap"
                ),
            },
            **extra,
        }
        atomic_json(self.status_path, value)
        return value

    def _save_state(self):
        atomic_json(self.state_path, self.state)

    def _snapshot_canonical(self):
        staging_root = self.root / "dataset_staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(
            prefix="dataset-", suffix=".parquet", dir=staging_root
        )
        try:
            with open(self.args.dataset, "rb") as source, os.fdopen(
                fd, "wb"
            ) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            return staged
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(staged)
            except OSError:
                pass
            raise

    def _inspect_and_publish_dataset(self):
        from checkpoint_orchestrator import inspect_dataset
        from pipeline.artifacts import GenerationStore

        staged = self._snapshot_canonical()
        try:
            _, _, strict, _ = inspect_dataset(
                staged,
                self.args.profile,
                self.args.solver_revision,
                self.args.library_revision,
            )
            rows = int(len(strict))
            store = GenerationStore(self.root / "artifacts")
            generation = store.publish_files(
                "experimental_dataset",
                {"train.parquet": staged},
                metadata={
                    "strict_full_rows": rows,
                    "solver_revision": self.args.solver_revision,
                    "library_revision": self.args.library_revision,
                    "source": os.path.abspath(self.args.dataset),
                },
            )
            return generation, rows
        finally:
            try:
                os.remove(staged)
            except FileNotFoundError:
                pass

    def _record_bootstrap_if_finished(self):
        path = self.args.bootstrap_status
        if not path or self.state.get("bootstrap_recorded"):
            return True
        status = _read_json(path)
        if not status:
            self.state["bootstrap_recorded"] = True
            self._save_state()
            return True
        supervisor_pid = status.get("supervisor_pid")
        phase = status.get("phase")
        if phase not in TERMINAL_PHASES and _alive(supervisor_pid):
            self._publish(
                "waiting_for_bootstrap_wave",
                bootstrap_status=os.path.abspath(path),
                bootstrap_phase=phase,
                bootstrap_supervisor_pid=supervisor_pid,
            )
            return False
        snapshot = status.get("strict_snapshot") or {}
        rows = snapshot.get("strict_full_rows")
        if rows is not None:
            self.state["last_attempted_strict_rows"] = int(rows)
            self.state["last_attempted_generation"] = snapshot.get(
                "source_dataset_sha256"
            )
        self.state.update({
            "bootstrap_recorded": True,
            "bootstrap_phase": phase,
            "bootstrap_recorded_at": _now(),
        })
        self._save_state()
        return True

    def _wave_commands(self, wave_root, dataset, generation_id):
        results = wave_root / "results"
        logs = wave_root / "logs"
        results.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        processes = []
        handles = []
        for target in TARGETS:
            result = results / f"{target}_lightgbm.json"
            stdout = open(logs / f"{target}.stdout.log", "ab")
            stderr = open(logs / f"{target}.stderr.log", "ab")
            handles.extend((stdout, stderr))
            command = [
                self.args.python,
                str(Path(self.args.code_root) / "training" / "tune_optuna.py"),
                "--experimental",
                "--target", target,
                "--family", "lightgbm",
                "--trials", str(self.args.trials),
                "--model-threads", str(self.args.hpo_threads),
                "--dataset", dataset,
                "--artifact-root", str(self.root / "artifacts"),
                "--result-json", str(result),
                "--solver-revision", self.args.solver_revision,
                "--library-revision", self.args.library_revision,
                "--data-contract-sha256", self.args.data_contract_sha256,
            ]
            process = subprocess.Popen(
                command,
                cwd=self.args.code_root,
                stdout=stdout,
                stderr=stderr,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
            )
            processes.append((target, process, result))
        return processes, handles

    def _launch_wave(self, generation, rows):
        generation_id = generation.generation_id
        wave_root = self.root / "waves" / f"wave-{generation_id[:16]}"
        wave_root.mkdir(parents=True, exist_ok=True)
        dataset = str(generation.path / "train.parquet")
        processes, handles = self._wave_commands(
            wave_root, dataset, generation_id
        )
        refresh_status = wave_root / "refresh" / "status.json"
        refresh_args = [
            self.args.python,
            str(Path(self.args.code_root) / "training" / "experimental_refresh.py"),
            "--runtime-root", str(wave_root / "refresh"),
            "--dataset", dataset,
            "--source-dataset-generation", f"dataset:{generation_id}",
            "--registry", self.args.experimental_registry,
            "--pointer", self.args.pointer,
            "--consumer-source", self.args.consumer_source,
            "--incumbent-source", self.args.consumer_source,
            "--profile", self.args.profile,
            "--thresholds", self.args.thresholds,
            "--solver-revision", self.args.solver_revision,
            "--library-revision", self.args.library_revision,
            "--model-threads", str(self.args.model_threads),
            "--target-workers", str(self.args.target_workers),
            "--trials-per-job", str(self.args.trials),
            "--hpo-threads-per-job", str(self.args.hpo_threads),
            "--poll-seconds", "10",
        ]
        for target, process, result in processes:
            refresh_args.extend([
                "--job", f"{target}|lightgbm|{process.pid}|{result}"
            ])
        refresh_stdout = open(wave_root / "refresh.stdout.log", "ab")
        refresh_stderr = open(wave_root / "refresh.stderr.log", "ab")
        handles.extend((refresh_stdout, refresh_stderr))
        refresh = subprocess.Popen(
            refresh_args,
            cwd=self.args.code_root,
            stdout=refresh_stdout,
            stderr=refresh_stderr,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        self.state.update({
            "active_wave": wave_root.name,
            "active_wave_status": str(refresh_status),
            "active_wave_dataset_generation": generation_id,
            "active_wave_strict_rows": rows,
            "active_wave_supervisor_pid": refresh.pid,
            "active_wave_started_at": _now(),
        })
        self._save_state()
        try:
            while refresh.poll() is None and not self.stop_path.exists():
                child_status = _read_json(refresh_status, {}) or {}
                self._publish(
                    "wave_running",
                    observed_strict_full_rows=rows,
                    next_refresh_strict_rows=rows + self.args.min_new_rows,
                    wave_supervisor_pid=refresh.pid,
                    wave_phase=child_status.get("phase", "starting"),
                )
                time.sleep(min(30, self.args.poll_seconds))
            if self.stop_path.exists() and refresh.poll() is None:
                refresh.terminate()
                refresh.wait(timeout=30)
            child_status = _read_json(refresh_status, {}) or {}
            if refresh.returncode == 0 and child_status.get("phase") in TERMINAL_PHASES:
                self.state.update({
                    "last_attempted_strict_rows": rows,
                    "last_attempted_generation": generation_id,
                    "last_result_phase": child_status.get("phase"),
                    "last_result": child_status.get("result"),
                    "last_finished_at": _now(),
                    "failure_count": 0,
                })
            else:
                self.state.update({
                    "failure_count": int(self.state.get("failure_count") or 0) + 1,
                    "last_failure_generation": generation_id,
                    "last_failure_at": _now(),
                    "last_failure": child_status.get("error")
                    or f"refresh_exit:{refresh.returncode}",
                })
        finally:
            for handle in handles:
                handle.close()
            self.state.update({
                "active_wave": None,
                "active_wave_status": None,
                "active_wave_supervisor_pid": None,
            })
            self._save_state()

    def run_forever(self):
        while not self.stop_path.exists():
            if not self._record_bootstrap_if_finished():
                time.sleep(self.args.poll_seconds)
                continue
            try:
                source = Path(self.args.dataset)
                observation = (
                    source.stat().st_size,
                    source.stat().st_mtime_ns,
                )
                if observation == self._last_observation:
                    self._publish("waiting_for_dataset_growth")
                    time.sleep(self.args.poll_seconds)
                    continue
                self._last_observation = observation
                generation, rows = self._inspect_and_publish_dataset()
                due, threshold = refresh_due(
                    rows,
                    self.state.get("last_attempted_strict_rows"),
                    self.args.min_new_rows,
                )
                duplicate = generation.generation_id == self.state.get(
                    "last_attempted_generation"
                )
                if not due or duplicate:
                    self._publish(
                        "waiting_for_dataset_growth",
                        observed_strict_full_rows=rows,
                        observed_dataset_generation=generation.generation_id,
                        next_refresh_strict_rows=threshold,
                        rows_until_refresh=max(0, threshold - rows),
                        duplicate_generation=duplicate,
                    )
                    time.sleep(self.args.poll_seconds)
                    continue
                if (
                    generation.generation_id
                    == self.state.get("last_failure_generation")
                    and int(self.state.get("failure_count") or 0) >= 3
                ):
                    self._publish(
                        "same_generation_failure_backoff",
                        observed_strict_full_rows=rows,
                        observed_dataset_generation=generation.generation_id,
                        failure_count=self.state.get("failure_count"),
                    )
                    time.sleep(max(300, self.args.poll_seconds))
                    continue
                self._launch_wave(generation, rows)
            except Exception as exc:
                self._publish(
                    "controller_error_retrying",
                    error=f"{type(exc).__name__}:{exc}",
                )
                time.sleep(max(30, self.args.poll_seconds))
        self._publish("stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--experimental-registry", required=True)
    parser.add_argument("--pointer", required=True)
    parser.add_argument("--consumer-source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--data-contract-sha256", required=True)
    parser.add_argument("--bootstrap-status", default=None)
    parser.add_argument("--min-new-rows", type=int, default=50)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--hpo-threads", type=int, default=3)
    parser.add_argument("--target-workers", type=int, default=4)
    parser.add_argument("--model-threads", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    for name in (
        "min_new_rows", "trials", "hpo_threads", "target_workers",
        "model_threads", "poll_seconds",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"{name} must be positive")
    for name in ("solver_revision", "library_revision"):
        value = str(getattr(args, name)).lower()
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            parser.error(f"{name} must be an exact revision")
        setattr(args, name, value)
    if (
        len(args.data_contract_sha256) != 64
        or any(char not in "0123456789abcdef" for char in args.data_contract_sha256)
    ):
        parser.error("data contract must be a SHA-256")
    args.dataset = os.path.abspath(args.dataset)
    args.code_root = os.path.abspath(args.code_root)
    lock_path = os.path.abspath(args.runtime_root) + ".lock"
    try:
        with FileLock(lock_path, timeout=0):
            ContinuousExperimentalRefresh(args).run_forever()
    except Timeout:
        raise SystemExit("experimental continuous controller is already running")


if __name__ == "__main__":
    main()
