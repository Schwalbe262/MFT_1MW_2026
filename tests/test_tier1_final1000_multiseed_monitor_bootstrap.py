from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools import tier1_final1000_multiseed_monitor as monitor
from tools import tier1_final1000_multiseed_monitor_bootstrap as bootstrap
from tools import tier1_final1000_multiseed_status as compact


def _backend(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    path = tmp_path / "fc51_successor_backend.py"
    path.write_text("COMPACT_V2 = True\n", encoding="utf-8")
    return {"regression_260707/monitoring/fc51_successor_backend.py": path}, path


def _identity(
    tmp_path: Path, bootstrap_manifest: Path, backend_files: dict[str, Path]
) -> dict:
    return {
        "bootstrap_manifest_path": bootstrap_manifest,
        "code_root": Path(bootstrap.__file__).parents[1],
        "code_revision": "a" * 40,
        "backend_files": backend_files,
        "backend_revision": "b" * 40,
        "test_revision": "c" * 40,
    }


def test_frozen_probe_is_deterministic_bounded_and_idempotent(tmp_path: Path):
    root = tmp_path / "probe"
    first = bootstrap.publish_frozen_probe(root)
    second = bootstrap.publish_frozen_probe(root)

    assert first["bootstrap_manifest_path"] == second["bootstrap_manifest_path"]
    assert first["index_file_sha256"] == second["index_file_sha256"]
    assert first["writes_performed"] > 0
    assert second["writes_performed"] == 0
    assert all(
        first[field] is False
        for field in (
            "scheduler_mutation_performed",
            "remote_write_performed",
            "aedt_used",
            "fea_submission_performed",
        )
    )

    frozen = bootstrap.load_frozen_probe(Path(first["bootstrap_manifest_path"]))
    page = monitor.adapt_condition_index(frozen["index_path"], limit=256)
    assert page["physical_lane_count"] == 2
    assert page["logical_seed_count"] == 5
    assert page["logical_sealed_seed_count"] == 3
    assert page["logical_unsealed_seed_count"] == 2
    assert page["hidden_unsealed_seed_record_count"] == 1
    assert sorted(record["seed"] for record in page["terminal_results"]) == [
        11001,
        12001,
        12002,
    ]
    assert sorted(item["task_id"] for item in page["latest_tasks"]) == [
        bootstrap.SINGLE_SEED_TASK_ID,
        bootstrap.BATCH4_TASK_ID,
    ]
    assert all(item["task_id"] > 0 for item in page["latest_tasks"])
    assert page["virtual_scheduler_task_ids_created"] is False

    for path in root.rglob("*"):
        if path.is_file():
            assert path.stat().st_size < compact.MAX_HOT_WIRE_BYTES
    index_path = Path(first["index_path"])
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() in index_path.name


def test_capability_binds_backend_adapter_generated_evidence_and_index(
    tmp_path: Path,
):
    published = bootstrap.publish_frozen_probe(tmp_path / "probe")
    manifest = Path(published["bootstrap_manifest_path"])
    backend_files, backend_path = _backend(tmp_path)
    identity = _identity(tmp_path, manifest, backend_files)
    sealed = bootstrap.seal_frozen_probe_capability(
        output_root=tmp_path / "capability", **identity
    )
    receipt_path = Path(sealed["receipt_path"])
    receipt = bootstrap.require_frozen_probe_capability(receipt_path, **identity)

    evidence_names = {
        item["logical_name"]
        for item in receipt["test_seals"]["evidence_identity"]["files"]
    }
    assert evidence_names == {
        "bootstrap/frozen-compact-v2-probe.json",
        "bootstrap/frozen-compact-v2-probe.junit.xml",
    }
    assert {
        item["logical_name"] for item in receipt["backend_identity"]["files"]
    } == set(backend_files)
    assert {item["logical_name"] for item in receipt["code_identity"]["files"]} == set(
        monitor.DEFAULT_ADAPTER_CODE_FILES
    )
    assert all(
        receipt[field] is False
        for field in (
            "scheduler_mutation_performed",
            "remote_write_performed",
            "aedt_used",
            "fea_submission_performed",
        )
    )

    backend_path.write_text("COMPACT_V2 = False\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend_identity drifted"):
        bootstrap.require_frozen_probe_capability(receipt_path, **identity)


def test_capability_refuses_probe_tamper_and_relocation(tmp_path: Path):
    published = bootstrap.publish_frozen_probe(tmp_path / "probe")
    manifest = Path(published["bootstrap_manifest_path"])
    backend_files, _backend_path = _backend(tmp_path)
    identity = _identity(tmp_path, manifest, backend_files)
    sealed = bootstrap.seal_frozen_probe_capability(
        output_root=tmp_path / "capability", **identity
    )
    receipt_path = Path(sealed["receipt_path"])

    moved_root = tmp_path / "moved-probe"
    shutil.copytree(manifest.parent, moved_root)
    moved_manifest = moved_root / manifest.name
    with pytest.raises(RuntimeError, match="relocated"):
        bootstrap.require_frozen_probe_capability(
            receipt_path,
            **{**identity, "bootstrap_manifest_path": moved_manifest},
        )

    evidence = Path(published["probe_evidence_path"])
    evidence.chmod(0o644)
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="probe evidence reference drifted"):
        bootstrap.require_frozen_probe_capability(receipt_path, **identity)


def test_capability_refuses_a_coherent_index_advance(tmp_path: Path):
    published = bootstrap.publish_frozen_probe(tmp_path / "probe")
    manifest = Path(published["bootstrap_manifest_path"])
    backend_files, _backend_path = _backend(tmp_path)
    identity = _identity(tmp_path, manifest, backend_files)
    sealed = bootstrap.seal_frozen_probe_capability(
        output_root=tmp_path / "capability", **identity
    )
    receipt_path = Path(sealed["receipt_path"])
    frozen = bootstrap.load_frozen_probe(manifest)

    status, shard_manifest, shards, index = compact.build_compact_snapshot(
        stage_id=bootstrap.FROZEN_STAGE_ID,
        physical_lanes=[
            {
                "task_id": bootstrap.BATCH4_TASK_ID,
                "protocol_version": bootstrap.PROTOCOL_VERSION,
                "state": "running",
                "batch_length": 4,
                "sealed_child_count": 3,
            }
        ],
        seed_records=[
            {
                "bundle_id": "advanced-batch4",
                "seed": seed,
                "physical_parent_task_id": bootstrap.BATCH4_TASK_ID,
                "batch_ordinal": ordinal,
            }
            for ordinal, seed in enumerate((13001, 13002, 13003))
        ],
        updated_at="2026-07-22T00:00:01+00:00",
    )
    advanced_root = tmp_path / "advanced"
    compact.publish_compact_snapshot(
        advanced_root, status, shard_manifest, shards, index
    )
    inputs = bootstrap._capability_inputs(**identity)
    assert inputs["index_path"] == frozen["index_path"]
    inputs["index_path"] = advanced_root / "index.json"
    with pytest.raises(RuntimeError, match="condition_index_identity drifted"):
        monitor.require_backend_capability(receipt_path, **inputs)


def test_bootstrap_cli_create_seal_and_validate(tmp_path: Path, capsys):
    root = tmp_path / "probe"
    assert bootstrap.main(["create", "--output-root", str(root)]) == 0
    created = json.loads(capsys.readouterr().out)
    backend_files, backend_path = _backend(tmp_path)
    backend_name = next(iter(backend_files))
    common = [
        "--bootstrap-manifest",
        created["bootstrap_manifest_path"],
        "--code-root",
        str(Path(bootstrap.__file__).parents[1]),
        "--code-revision",
        "a" * 40,
        "--backend-file",
        f"{backend_name}={backend_path}",
        "--backend-revision",
        "b" * 40,
        "--test-revision",
        "c" * 40,
    ]
    assert (
        bootstrap.main(
            ["seal-capability", "--output-root", str(tmp_path / "caps"), *common]
        )
        == 0
    )
    sealed = json.loads(capsys.readouterr().out)
    assert (
        bootstrap.main(
            ["validate-capability", "--receipt", sealed["receipt_path"], *common]
        )
        == 0
    )
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["receipt_sha256"] == sealed["receipt_sha256"]
