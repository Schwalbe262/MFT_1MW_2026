"""Process supervisor for independent collector/train/tune/optimize/FEA lanes."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
import uuid

from .artifacts import GenerationStore
from .queue import DurableJobQueue
from .runner import JobRunner


DEFAULT_LANES = {
    "collector": (["collect"], 1),
    "trainer": (["train"], 1),
    "tuner": (["tune"], 1),
    "optimizer": (["optimize"], 1),
    "standard-verifier": (["verify_standard"], 1),
    "fine-verifier": (["verify_fine"], 1),
}


def _worker_main(queue_path, artifact_root, work_root, lane, job_types, stop):
    runner = JobRunner(
        DurableJobQueue(queue_path),
        GenerationStore(artifact_root),
        work_root,
        owner=f"{lane}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
    )
    runner.run_forever(job_types, stop_event=stop)


class PipelineSupervisor:
    def __init__(self, queue_path, artifact_root, work_root, lanes=None):
        self.queue_path = os.path.abspath(queue_path)
        self.artifact_root = os.path.abspath(artifact_root)
        self.work_root = os.path.abspath(work_root)
        self.lanes = dict(lanes or DEFAULT_LANES)

    def run(self) -> None:
        stop = multiprocessing.Event()
        children = []

        def request_stop(*_):
            stop.set()

        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), request_stop)
        for lane, (types, count) in self.lanes.items():
            for index in range(int(count)):
                process = multiprocessing.Process(
                    target=_worker_main,
                    args=(
                        self.queue_path,
                        self.artifact_root,
                        self.work_root,
                        f"{lane}-{index}",
                        list(types),
                        stop,
                    ),
                    name=f"mft-pipeline-{lane}-{index}",
                )
                process.start()
                children.append(process)
        try:
            while not stop.is_set():
                for process in children:
                    if not process.is_alive() and process.exitcode is not None:
                        stop.set()
                        raise RuntimeError(
                            f"pipeline lane {process.name} exited unexpectedly "
                            f"with code {process.exitcode}"
                        )
                stop.wait(2.0)
        finally:
            stop.set()
            deadline = time.monotonic() + 30
            for process in children:
                process.join(max(0.0, deadline - time.monotonic()))
            for process in children:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
