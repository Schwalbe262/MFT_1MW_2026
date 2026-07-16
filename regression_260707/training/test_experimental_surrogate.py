import hashlib
import json
from pathlib import Path
import tempfile
import pandas as pd

from regression_260707.training import experimental_refresh as refresh
from regression_260707.training import experimental_surrogate as experimental


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _generation(registry, run_id, dataset_sha, loss, rows):
    generation = registry / "generations" / run_id
    generation.mkdir(parents=True)
    artifact = generation / "model.bin"
    artifact.write_bytes(run_id.encode("ascii"))
    report = {
        "training_run_id": run_id,
        "dataset_sha256": dataset_sha,
        "strict_full_rows": rows,
        "profile_sha256": "a" * 64,
        "artifacts": {
            "model.bin": hashlib.sha256(artifact.read_bytes()).hexdigest()
        },
        "report": {"Llt_phys": {"mape_pct": loss}},
    }
    _write_json(generation / "train_report.json", report)
    return generation, report


def _quality(path, generation, dataset_sha, rows):
    report_sha = experimental.sha256_file(generation / "train_report.json")
    value = {
        "schema_version": 1,
        "lane": "provisional_2000_surrogate",
        "passed": False,
        "strict_full_rows": rows,
        "dataset_sha256": dataset_sha,
        "generation_report_sha256": report_sha,
        "failed_targets": {"Llt_phys": ["mape_pct"]},
        "solver_revision_pin": "1" * 40,
        "library_revision_pin": "2" * 40,
    }
    _write_json(path, value)
    return value


def test_better_candidate_updates_audit_pointer_and_nsga_source_identically():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = root / "registry"
        dataset = root / "train.parquet"
        dataset.write_bytes(b"dataset")
        dataset_sha = experimental.sha256_file(dataset)
        incumbent_generation, _ = _generation(
            registry, "incumbent", dataset_sha, 1.0, 2100
        )
        candidate_generation, _ = _generation(
            registry, "candidate", dataset_sha, 0.8, 2200
        )
        incumbent_quality = root / "incumbent_quality.json"
        _quality(incumbent_quality, incumbent_generation, dataset_sha, 2100)
        source = root / "experimental_source.json"
        _write_json(source, {
            "schema_version": experimental.MODEL_SOURCE_SCHEMA,
            "lane": "experimental",
            "production_eligible": False,
            "fea_submission_approved": False,
            "registry": str(registry),
            "generation": str(incumbent_generation),
            "dataset": str(dataset),
            "quality_status": str(incumbent_quality),
            "fea_solver_revision": "1" * 40,
            "fea_library_revision": "2" * 40,
        })
        candidate_quality = _quality(
            root / "candidate_quality.input.json",
            candidate_generation,
            dataset_sha,
            2200,
        )
        pointer = root / "experimental_surrogate.json"

        result = experimental.publish_if_better(
            registry=registry,
            generation=candidate_generation,
            dataset=dataset,
            quality=candidate_quality,
            evidence_root=root / "evidence",
            pointer_path=pointer,
            consumer_path=source,
            incumbent_source=source,
            solver_revision="1" * 40,
            library_revision="2" * 40,
        )

        assert result["promoted"] is True
        assert pointer.read_bytes() == source.read_bytes()
        active = experimental.validate_source_descriptor(source)
        assert active["record"]["report"]["training_run_id"] == "candidate"
        assert active["source"]["eligibility"] == "FEA-NOT-APPROVED"


def test_candidate_snapshot_contains_only_exact_pinned_strict_rows():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "train.parquet"
        source.write_bytes(b"immutable source identity")
        audited = pd.DataFrame({
            "physics_data_revision": ["v3", "legacy_unspecified"],
            "value": [1.0, 2.0],
            "_strict_valid_em": [True, False],
            "_strict_valid_thermal": [True, False],
            "_strict_valid_full": [True, False],
            "_strict_invalid_reasons": ["", "revision_mismatch"],
        })
        strict = audited.iloc[[0]].copy()
        snapshot, evidence = refresh._strict_snapshot(
            source, root / "work", root / "profile.json",
            "1" * 40, "2" * 40,
            inspector=lambda *_: (
                audited, audited, strict, {"revision_mismatch": 1}
            ),
        )

        candidate = pd.read_parquet(snapshot)
        assert candidate["physics_data_revision"].tolist() == ["v3"]
        assert not any(column.startswith("_strict_") for column in candidate)
        assert evidence["strict_full_rows"] == 1
        assert evidence["solver_revision"] == "1" * 40
        assert evidence["snapshot_sha256"] == experimental.sha256_file(snapshot)
