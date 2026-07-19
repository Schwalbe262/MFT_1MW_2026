from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import time
from unittest import mock

import pandas as pd
import pytest

from regression_260707.training import train_models


def test_target_workers_train_independent_targets_concurrently():
    targets = ["Llt_phys", "k", "P_winding_total", "B_max_core"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "strict.parquet"
        dataset.write_bytes(b"strict cohort")
        profile = root / "profile.json"
        profile.write_text('{"param_overrides": {}}', encoding="utf-8")
        args = SimpleNamespace(
            registry=str(root / "registry"),
            dataset=str(dataset),
            profile=str(profile),
            weight_col=None,
            min_rows=1,
            params=None,
            source_dataset_path=str(dataset),
            source_dataset_generation="dataset:" + "a" * 64,
            target_workers=4,
            model_threads=2,
            max_model_thread_budget=8,
        )
        frame = pd.DataFrame({target: [1.0] for target in targets})
        frame["feature"] = 1.0
        recovery_audit = {
            "contract": "mft-capacitance-lc-inverse-v1",
            "recovered_row_count": 7212,
            "max_observed_abs_delta_F": 4.999e-11,
        }
        frame.attrs["capacitance_recovery"] = recovery_audit
        active = 0
        maximum = 0
        lock = threading.Lock()

        def fake_train(_frame, _features, target, *_args, **_kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return ({
                "models": [],
                "physics_data_revision_cohort": "v3",
            }, {
                "r2": 0.99,
                "mape_pct": 1.0,
                "p90_ape_pct": 2.0,
                "interval_coverage": 0.9,
            })

        with mock.patch.object(train_models, "train_target", side_effect=fake_train):
            result = train_models._build_candidate(
                args,
                frame,
                ["feature"],
                2200,
                targets,
                lambda _target: {},
            )

        report = json.loads(
            (Path(result["generation_path"]) / "train_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert maximum >= 2
        assert set(report["report"]) == set(targets)
        assert set(report["target_physics_data_revision_cohorts"].values()) == {"v3"}
        assert report["training_parallelism"] == {
            "target_workers": 4,
            "model_threads": 2,
            "maximum_model_thread_budget": 8,
            "effective_model_thread_budget": 8,
        }
        assert report["capacitance_recovery"] == recovery_audit
        assert result["capacitance_recovery"] == recovery_audit
        for target in targets:
            metadata = json.loads(
                (
                    Path(result["generation_path"]) / target / "meta.json"
                ).read_text(encoding="utf-8")
            )
            assert metadata["capacitance_recovery"] == recovery_audit


def test_target_workers_cannot_oversubscribe_declared_budget():
    argv = [
        "train_models.py",
        "--model-threads", "3",
        "--target-workers", "4",
        "--max-model-thread-budget", "8",
    ]
    with mock.patch("sys.argv", argv):
        with pytest.raises(SystemExit):
            train_models.main()
