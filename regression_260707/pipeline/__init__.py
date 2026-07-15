"""Durable, generation-pinned orchestration for the MFT design loop.

The package is intentionally independent from the scheduler implementation.
It coordinates the existing collector, checkpoint trainer, optimizer, and
verification entrypoints while keeping every hand-off content addressed.
"""

from .artifacts import GenerationStore, PublishedGeneration
from .queue import DurableJobQueue, Job

__all__ = [
    "DurableJobQueue",
    "GenerationStore",
    "Job",
    "PublishedGeneration",
]
