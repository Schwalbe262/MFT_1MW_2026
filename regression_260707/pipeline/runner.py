"""Lease-aware subprocess workers for the durable pipeline queue."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from .artifacts import GenerationStore
from .queue import DurableJobQueue, Job


class LeaseLost(RuntimeError):
    pass


class _LeaseKeeper:
    """Renew one queue lease until every output byte is authenticated."""

    def __init__(self, queue, job_id, owner, lease_seconds):
        self.queue = queue
        self.job_id = int(job_id)
        self.owner = owner
        self.lease_seconds = float(lease_seconds)
        self.interval = max(1.0, self.lease_seconds / 4.0)
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"pipeline-lease-{self.job_id}",
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval + 1.0))

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                renewed = self.queue.heartbeat(
                    self.job_id,
                    self.owner,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:  # fail closed; another worker may recover
                self.error = exc
                renewed = False
            if not renewed:
                self.lost_event.set()
                return

    @property
    def lost(self):
        return self.lost_event.is_set()

    def ensure_owned(self):
        if self.lost:
            detail = f":{type(self.error).__name__}:{self.error}" if self.error else ""
            raise LeaseLost(f"queue lease was lost{detail}")


def _replace_tokens(value, replacements: dict[str, str]):
    if isinstance(value, str):
        for token, replacement in replacements.items():
            value = value.replace("{" + token + "}", replacement)
        return value
    if isinstance(value, list):
        return [_replace_tokens(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_tokens(item, replacements)
            for key, item in value.items()
        }
    return value


class JobRunner:
    """Execute one command payload while renewing the queue ownership lease."""

    def __init__(
        self,
        queue: DurableJobQueue,
        store: GenerationStore,
        work_root: str | os.PathLike[str],
        *,
        owner: str | None = None,
        lease_seconds: float = 120.0,
    ):
        self.queue = queue
        self.store = store
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.owner = owner or f"worker-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = float(lease_seconds)
        if self.lease_seconds < 15:
            raise ValueError("worker lease must be at least 15 seconds")

    def run_once(
        self, job_types: list[str] | None = None, *, stop_event=None
    ) -> Job | None:
        job = self.queue.claim(
            self.owner,
            job_types=job_types,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return None
        return self._execute(job, stop_event=stop_event)

    def _execute(self, job: Job, stop_event=None) -> Job:
        with _LeaseKeeper(
            self.queue, job.id, self.owner, self.lease_seconds
        ) as lease:
            return self._execute_owned(job, lease, stop_event=stop_event)

    def _execute_owned(self, job: Job, lease: _LeaseKeeper, stop_event=None) -> Job:
        work_dir = self.work_root / f"job-{job.id:08d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        replacements = {
            "job_id": str(job.id),
            "attempt": str(job.attempt),
            "work_dir": str(work_dir),
        }
        dependency_type_counts: dict[str, int] = {}
        dependency_kinds = job.payload.get("dependency_kinds") or {}
        if not isinstance(dependency_kinds, dict):
            return self.queue.fail(
                job.id, self.owner, "invalid dependency kind contract", retry=False
            )
        for dependency in self.queue.dependencies(job.id):
            if dependency.state != "succeeded" or not dependency.output_generation:
                return self.queue.fail(
                    job.id,
                    self.owner,
                    f"dependency_output_unavailable:{dependency.id}",
                    retry=False,
                )
            try:
                authenticated = self.store.load(dependency.output_generation)
            except Exception as exc:
                return self.queue.fail(
                    job.id,
                    self.owner,
                    f"dependency_output_authentication_failed:{dependency.id}:"
                    f"{type(exc).__name__}:{exc}",
                    retry=False,
                )
            expected_kind = dependency_kinds.get(dependency.job_type)
            if not expected_kind:
                return self.queue.fail(
                    job.id,
                    self.owner,
                    f"dependency_output_kind_contract_missing:{dependency.id}",
                    retry=False,
                )
            if authenticated.kind != str(expected_kind):
                return self.queue.fail(
                    job.id,
                    self.owner,
                    f"dependency_output_kind_mismatch:{dependency.id}:"
                    f"{authenticated.kind}!={expected_kind}",
                    retry=False,
                )
            replacements[f"dependency_{dependency.id}_output"] = str(
                authenticated.path
            )
            dependency_type_counts[dependency.job_type] = (
                dependency_type_counts.get(dependency.job_type, 0) + 1
            )
            replacements[f"dependency_{dependency.job_type}_output"] = (
                str(authenticated.path)
            )
        for job_type, count in dependency_type_counts.items():
            if count > 1:
                replacements.pop(f"dependency_{job_type}_output", None)
        payload = _replace_tokens(job.payload, replacements)
        command = payload.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) and value for value in command)
        ):
            return self.queue.fail(
                job.id, self.owner, "invalid command payload", retry=False
            )
        cwd = os.path.abspath(payload.get("cwd") or os.getcwd())
        environment = os.environ.copy()
        raw_environment = payload.get("env") or {}
        if not isinstance(raw_environment, dict):
            return self.queue.fail(
                job.id, self.owner, "invalid environment payload", retry=False
            )
        raw_non_retryable_codes = payload.get("non_retryable_exit_codes", [])
        if (
            not isinstance(raw_non_retryable_codes, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > 255
                for value in raw_non_retryable_codes
            )
        ):
            return self.queue.fail(
                job.id,
                self.owner,
                "invalid non_retryable_exit_codes payload",
                retry=False,
            )
        non_retryable_exit_codes = set(raw_non_retryable_codes)
        environment.update({str(key): str(value) for key, value in raw_environment.items()})
        log_path = work_dir / f"attempt-{job.attempt:03d}.log"
        process = None
        lease_lost = False
        shutdown_requested = False
        try:
            with open(log_path, "ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=(os.name != "nt"),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    ),
                )
                while process.poll() is None:
                    if stop_event is not None and stop_event.is_set():
                        shutdown_requested = True
                        self._terminate(process)
                        break
                    if lease.lost:
                        lease_lost = True
                        self._terminate(process)
                        break
                    if stop_event is not None:
                        stop_event.wait(0.5)
                    else:
                        time.sleep(0.5)
                return_code = process.wait()
            if lease_lost:
                latest = self.queue.get(job.id)
                return latest if latest is not None else job
            if shutdown_requested:
                return self.queue.fail(
                    job.id,
                    self.owner,
                    "worker_shutdown",
                    retry=True,
                    base_backoff_seconds=5,
                )
            if return_code != 0:
                deterministic_terminal = return_code in non_retryable_exit_codes
                return self.queue.fail(
                    job.id,
                    self.owner,
                    (
                        f"command_exit:{return_code};"
                        f"non_retryable={str(deterministic_terminal).lower()};"
                        f"log={log_path}"
                    ),
                    retry=(
                        bool(payload.get("retry", True))
                        and not deterministic_terminal
                    ),
                    base_backoff_seconds=float(
                        payload.get("retry_backoff_seconds", 30)
                    ),
                )

            output_generation = None
            publish = payload.get("publish")
            if publish is not None:
                lease.ensure_owned()
                if not isinstance(publish, dict):
                    raise ValueError("publish payload must be an object")
                kind = str(publish["kind"])
                source = Path(str(publish["source"])).resolve()
                metadata = dict(publish.get("metadata") or {})
                metadata.update(
                    queue_job_id=job.id,
                    queue_job_type=job.job_type,
                    input_generation=job.input_generation,
                )
                parents = list(publish.get("parents") or [])
                if job.input_generation:
                    parents.append(job.input_generation)
                generation = self.store.publish_tree(
                    kind, source, metadata=metadata, parents=parents
                )
                lease.ensure_owned()
                output_generation = str(generation.path)
            result_json = payload.get("result_json")
            if result_json and output_generation is None:
                lease.ensure_owned()
                result = json.loads(Path(result_json).read_text(encoding="utf-8"))
                key = str(payload.get("result_output_key") or "generation_path")
                value = result
                for component in key.split("."):
                    value = value[component]
                output_generation = os.path.abspath(os.fspath(value))
                expected_kind = payload.get("result_generation_kind")
                if not expected_kind:
                    raise ValueError(
                        "result_json output requires result_generation_kind"
                    )
                generation = self.store.load(output_generation)
                if generation.kind != str(expected_kind):
                    raise ValueError(
                        f"result generation kind mismatch: "
                        f"{generation.kind}!={expected_kind}"
                    )
                id_key = str(
                    payload.get("result_generation_id_key") or "generation_id"
                )
                expected_id = result
                for component in id_key.split("."):
                    expected_id = expected_id[component]
                if str(expected_id) != generation.generation_id:
                    raise ValueError("result generation identity mismatch")
                output_generation = str(generation.path)
                lease.ensure_owned()
            lease.ensure_owned()
            return self.queue.succeed(
                job.id,
                self.owner,
                output_generation=output_generation,
            )
        except Exception as exc:
            if lease_lost or lease.lost or isinstance(exc, LeaseLost):
                latest = self.queue.get(job.id)
                return latest if latest is not None else job
            try:
                return self.queue.fail(
                    job.id,
                    self.owner,
                    f"runner_error:{type(exc).__name__}:{exc}",
                    retry=bool(payload.get("retry", True)),
                    base_backoff_seconds=float(
                        payload.get("retry_backoff_seconds", 30)
                    ),
                )
            except RuntimeError:
                latest = self.queue.get(job.id)
                return latest if latest is not None else job
        finally:
            if process is not None and process.poll() is None:
                self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=20,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except Exception:
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass

    def run_forever(
        self,
        job_types: list[str] | None = None,
        *,
        poll_seconds: float = 5.0,
        stop_event=None,
    ) -> None:
        while stop_event is None or not stop_event.is_set():
            if self.run_once(job_types, stop_event=stop_event) is None:
                if stop_event is not None:
                    stop_event.wait(poll_seconds)
                else:
                    time.sleep(poll_seconds)
