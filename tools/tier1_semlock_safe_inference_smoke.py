"""Fail-closed smoke for semaphore-free Tier-1 surrogate prediction.

Run this inside an immutable remote bundle before admitting its tasks::

    python tools/tier1_semlock_safe_inference_smoke.py \
      --code-root . --model-bundle ARTIFACT/models.pkl \
      --expected-sha256 SHA256 --threads 4 --repeats 32

The command never contacts the Scheduler.  It loads one authenticated model
artifact, forbids joblib ``ThreadPool`` and multiprocessing ``SemLock``
construction, and checks that repeated predictions do not grow the process
thread or file-descriptor inventory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Callable
from unittest import mock

import numpy as np
import pandas as pd


SCHEMA_VERSION = "mft-tier1-semlock-safe-inference-smoke-v1"
SKLEARN_FOREST_FAMILIES = {"extratrees", "randomforest"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def process_resource_snapshot() -> dict[str, int | None]:
    """Return process-owned resource counts without optional dependencies."""

    def count(path: str) -> int | None:
        try:
            return len(os.listdir(path))
        except OSError:
            return None

    linux_threads = count("/proc/self/task")
    if linux_threads is None:
        import threading

        linux_threads = threading.active_count()
    return {
        "process_threads": linux_threads,
        "process_file_descriptors": count("/proc/self/fd"),
    }


def _prediction_tuple(predictor: Any, frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    mean, half_width = predictor.predict_mu_sigma(frame, conformal=True)
    disagreement = predictor.disagreement(frame)
    return tuple(
        np.asarray(values, dtype=float).copy()
        for values in (mean, half_width, disagreement)
    )


def attest_predictor(
    predictor: Any,
    frame: pd.DataFrame,
    *,
    threads: int = 4,
    repeats: int = 32,
    resource_probe: Callable[[], dict[str, int | None]] = (
        process_resource_snapshot
    ),
) -> dict[str, Any]:
    """Stress one predictor while denying every known SemLock creation path."""

    if isinstance(threads, bool) or not 1 <= int(threads) <= 8:
        raise ValueError("threads must be an integer from 1 through 8")
    if isinstance(repeats, bool) or int(repeats) < 2:
        raise ValueError("repeats must be an integer of at least two")
    threads = int(threads)
    repeats = int(repeats)
    binding = predictor.configure_inference_threads(threads)
    if not isinstance(binding, dict):
        raise RuntimeError("predictor inference binding is unavailable")
    family_threads = binding.get("family_threads")
    configured_families = binding.get("families")
    if (
        not isinstance(family_threads, dict)
        or not isinstance(configured_families, list)
        or set(family_threads) != set(configured_families)
        or any(
            family_threads[family]
            != (1 if family in SKLEARN_FOREST_FAMILIES else threads)
            for family in configured_families
        )
    ):
        raise RuntimeError("unsafe family-specific inference binding")
    families = sorted(set(configured_families))

    import _multiprocessing
    import joblib._parallel_backends as parallel_backends

    forbidden_calls = {"joblib_threadpool": 0, "multiprocessing_semlock": 0}

    def forbid_threadpool(*_args: Any, **_kwargs: Any) -> Any:
        forbidden_calls["joblib_threadpool"] += 1
        raise RuntimeError("joblib ThreadPool construction is forbidden")

    def forbid_semlock(*_args: Any, **_kwargs: Any) -> Any:
        forbidden_calls["multiprocessing_semlock"] += 1
        raise RuntimeError("multiprocessing SemLock construction is forbidden")

    with mock.patch.object(
        parallel_backends, "ThreadPool", new=forbid_threadpool
    ), mock.patch.object(_multiprocessing, "SemLock", new=forbid_semlock):
        reference = _prediction_tuple(predictor, frame)
        before = resource_probe()
        for _ in range(repeats - 1):
            observed = _prediction_tuple(predictor, frame)
            if any(
                not np.array_equal(expected, actual)
                for expected, actual in zip(reference, observed)
            ):
                raise RuntimeError("repeated surrogate prediction drifted")
        gc.collect()
        after = resource_probe()

    growth = {
        name: (
            max(int(after[name]) - int(before[name]), 0)
            if before.get(name) is not None and after.get(name) is not None
            else None
        )
        for name in sorted(set(before) | set(after))
    }
    if any(value not in {None, 0} for value in growth.values()):
        raise RuntimeError(f"prediction resource inventory grew: {growth}")
    if any(forbidden_calls.values()):
        raise RuntimeError("forbidden parallel primitive was constructed")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "threads": threads,
        "repeats": repeats,
        "rows": int(len(frame)),
        "model_count": int(binding.get("model_count") or 0),
        "families": families,
        "family_threads": dict(sorted(family_threads.items())),
        "semaphore_free_families": sorted(
            set(families) & SKLEARN_FOREST_FAMILIES
        ),
        "forbidden_constructor_calls": forbidden_calls,
        "resource_before": before,
        "resource_after": after,
        "positive_resource_growth": growth,
        "repeated_prediction_bitwise_equal": True,
    }


def _load_predictor(code_root: Path, model_bundle: Path) -> Any:
    training = code_root / "regression_260707" / "training"
    regression = code_root / "regression_260707"
    for path in (code_root, regression, training):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    module = importlib.import_module("predictor")
    if Path(module.__file__).resolve() != (training / "predictor.py").resolve():
        raise RuntimeError("predictor import escaped the authenticated code root")
    with model_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    return module.EnsemblePredictor(bundle)


def _frame(features: list[str], rows: int) -> pd.DataFrame:
    values = np.linspace(
        -0.25,
        0.25,
        num=rows * len(features),
        dtype=float,
    ).reshape(rows, len(features))
    return pd.DataFrame(values, columns=features)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    code_root = args.code_root.resolve(strict=True)
    model_bundle = args.model_bundle.resolve(strict=True)
    expected = args.expected_sha256.strip().lower()
    observed = _sha256(model_bundle)
    if len(expected) != 64 or observed != expected:
        raise RuntimeError("model bundle SHA256 mismatch")
    if args.rows < 1:
        raise ValueError("rows must be positive")

    predictor = _load_predictor(code_root, model_bundle)
    receipt = attest_predictor(
        predictor,
        _frame(list(predictor.features), int(args.rows)),
        threads=args.threads,
        repeats=args.repeats,
    )
    receipt["code_root"] = str(code_root)
    receipt["model_bundle"] = str(model_bundle)
    receipt["model_bundle_sha256"] = observed
    receipt["sha256"] = _canonical_sha256(receipt)
    payload = json.dumps(
        receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
