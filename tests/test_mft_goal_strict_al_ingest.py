from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from module.core_material_contract import PHYSICS_DATA_REVISION
from module.input_parameter_260706 import create_input_parameter
from module.mft_goal_20260726_contract import (
    GOAL_G0_MODEL_TARGETS,
    canonical_sha256,
)
from regression_260707.quality_contract import annotate_validity
from regression_260707.test_pipeline_completion import _valid_native_result
from regression_260707.training.checkpoint_train import to_physical
from tools import mft_goal_strict_al_ingest as ingest


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _profile() -> dict:
    return json.loads(ingest.GOAL_PROFILE_PATH.read_text(encoding="utf-8"))


def _strict_row(*, task_id: int, n1: int, **updates) -> dict:
    params = create_input_parameter({}).iloc[0].to_dict()
    params.update(_profile()["param_overrides"])
    capacitances = {
        "C_tx_tx_F": 80e-12,
        "C_rx_rx_F": 32e-12,
        "C_tx_rx_F": 16e-12,
    }
    inductances = {
        "cap_L_tx_self_H": 200e-6,
        "cap_L_rx_self_H": 800e-6,
        "cap_L_leakage_H": 20e-6,
    }
    frequencies = {
        "f_res_tx_self_Hz": 1.0 / (
            2.0
            * math.pi
            * math.sqrt(
                inductances["cap_L_tx_self_H"]
                * capacitances["C_tx_tx_F"]
            )
        ),
        "f_res_rx_self_Hz": 1.0 / (
            2.0
            * math.pi
            * math.sqrt(
                inductances["cap_L_rx_self_H"]
                * capacitances["C_rx_rx_F"]
            )
        ),
        "f_res_interwinding_Hz": 1.0 / (
            2.0
            * math.pi
            * math.sqrt(
                inductances["cap_L_leakage_H"]
                * capacitances["C_tx_rx_F"]
            )
        ),
    }
    params.update({
        **capacitances,
        **inductances,
        **frequencies,
        "thermal_pad_conductivity_W_mK": 0.2,
        "N1": n1,
        "task_id": task_id,
        "project_name": f"strict-project-{task_id}",
        "saved_at": f"2026-07-25 12:{task_id % 60:02d}:00",
        "git_hash": SOLVER_REVISION,
        "pyaedt_library_git_hash": LIBRARY_REVISION,
        "physics_data_revision": PHYSICS_DATA_REVISION,
    })
    params.update(updates)
    return _valid_native_result(**params)


def _base_dataset(tmp_path: Path, row_count: int = 8) -> Path:
    rows = [
        _strict_row(task_id=100 + index, n1=5 + index % 4)
        for index in range(row_count)
    ]
    frame = annotate_validity(pd.DataFrame(rows), _profile())
    frame = to_physical(frame)
    path = tmp_path / "base.parquet"
    frame.to_parquet(path, index=False)
    return path


def _truth(
    tmp_path: Path,
    *,
    task_id: int,
    n1: int,
    source_task: int | None = None,
) -> ingest.AuthenticatedTruth:
    result = _strict_row(task_id=task_id, n1=n1)
    collection_path = tmp_path / f"collection-{task_id}.json"
    collection_path.write_text(
        json.dumps({"fixture": task_id}), encoding="utf-8"
    )
    collection = {
        "schema_version": ingest.PRODUCTION_COLLECTION_SCHEMA,
        "payload_sha256": _sha(f"collection-{task_id}"),
        "result_sha256": canonical_sha256(result),
        "candidate_physics_sha256": _sha(f"candidate-{task_id}"),
        "task_id": task_id,
        "goal_physical_spec_passed": False,
    }
    plan = {
        "payload_sha256": _sha(f"plan-{task_id}"),
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
    }
    return ingest.AuthenticatedTruth(
        adapter_kind="production",
        collection_path=collection_path,
        collection_file_sha256=ingest._sha256_file(collection_path),
        collection=collection,
        plan=plan,
        result=result,
        solver_revision=SOLVER_REVISION,
        library_revision=LIBRARY_REVISION,
        source_task_payload_sha256=_sha(
            f"source-task-{source_task if source_task is not None else task_id}"
        ),
        source_seed=task_id,
        source_fixed_primary_turns=n1,
    )


def _prepare(
    tmp_path: Path,
    monkeypatch,
    *,
    new_count: int = 8,
    source_task_count: int | None = None,
) -> tuple[Path, ingest.PreparedIngest]:
    base = _base_dataset(tmp_path)
    paths = [
        tmp_path / f"collection-{1000 + index}.json"
        for index in range(new_count)
    ]
    truths = {
        path.resolve(): _truth(
            tmp_path,
            task_id=1000 + index,
            n1=6,
            source_task=(
                index
                if source_task_count is None
                else index % source_task_count
            ),
        )
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        ingest,
        "authenticate_collection",
        lambda path: truths[path.resolve()],
    )
    prepared = ingest.prepare_ingest(
        base_dataset=base,
        expected_base_sha256=ingest._sha256_file(base),
        expected_base_rows=8,
        collection_paths=paths,
    )
    return base, prepared


def test_build_isolated_dataset_adds_all_targets_and_preserves_base(
    tmp_path, monkeypatch
):
    base, prepared = _prepare(tmp_path, monkeypatch)
    before = ingest._sha256_file(base)

    assert prepared.retraining_admission["allowed"] is True
    assert prepared.retraining_admission["strict_new_rows"] == 8
    assert prepared.retraining_admission["unique_complete_geometries"] == 8
    assert (
        prepared.retraining_admission[
            "minimum_unique_complete_geometries"
        ]
        == 8
    )
    assert prepared.retraining_admission["unique_source_tasks"] == 8
    assert prepared.retraining_admission["targeted_strata"] == [6]
    assert prepared.retraining_admission["global_N1_coverage_claimed"] is False
    assert prepared.retraining_admission[
        "new_NSGA2_campaign_all_N1_strata_required"
    ] is True
    for target in GOAL_G0_MODEL_TARGETS:
        assert (
            prepared.output_target_rows[target]
            == prepared.base_target_rows[target] + 8
        )

    output = tmp_path / "isolated-output"
    manifest_path = ingest.build_immutable_bundle(
        prepared,
        output_dir=output,
        base_dataset=base,
        repository_identity={
            "revision": "c" * 40,
            "dirty": False,
            "tool_source": {
                "path": "fixture",
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
        },
    )

    assert ingest._sha256_file(base) == before
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    observed_seal = unsigned.pop("payload_sha256")
    assert observed_seal == canonical_sha256(unsigned)
    assert manifest["canonical_source_mutated"] is False
    assert manifest["scheduler_submission_performed"] is False
    assert manifest["new_model_generation_required"] is True
    assert manifest["old_generation_result_mixing_allowed"] is False
    assert manifest["next_campaign_contract"] == {
        "seed_start": 2607263000,
        "seed_end_inclusive": 2607263511,
        "seed_count": 512,
        "all_four_N1_strata_required": True,
        "single_dataset_sha256_required": manifest["output_dataset"]["sha256"],
        "single_model_generation_required": True,
        "old_generation_result_mixing_allowed": False,
    }
    output_frame = pd.read_parquet(output / "strict_al.parquet")
    assert len(output_frame) == 16
    assert output_frame["goal_al_authenticated_standard_truth"].notna().sum() == 8

    with pytest.raises(ingest.StrictALIngestError, match="already exists"):
        ingest.build_immutable_bundle(
            prepared,
            output_dir=output,
            base_dataset=base,
            repository_identity={"revision": "c" * 40, "dirty": False},
        )


def test_small_valid_batch_is_not_retraining_ready(tmp_path, monkeypatch):
    _base, prepared = _prepare(tmp_path, monkeypatch, new_count=4)

    assert prepared.retraining_admission["allowed"] is False
    assert prepared.retraining_admission["strict_new_rows"] == 4
    assert "strict_new_rows<8" in prepared.retraining_admission["reasons"]


def test_eight_rows_from_only_three_source_tasks_are_not_ready(
    tmp_path, monkeypatch
):
    _base, prepared = _prepare(
        tmp_path,
        monkeypatch,
        new_count=8,
        source_task_count=3,
    )

    assert prepared.retraining_admission["allowed"] is False
    assert prepared.retraining_admission["unique_source_tasks"] == 3
    assert "unique_source_tasks<4" in prepared.retraining_admission["reasons"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"fan_velocity": 2.0}, "fixed operating/cooling identity"),
        ({"core_plate_pad_t": 3.0}, "fixed operating/cooling identity"),
        ({"wcp_pad_t": 3.0}, "fixed operating/cooling identity"),
        ({"k_ins": 0.3}, "fixed operating/cooling identity"),
        (
            {"thermal_pad_conductivity_W_mK": 0.3},
            "fixed operating/cooling identity",
        ),
        ({"git_dirty": 1}, "strict EM/thermal validity"),
        ({"C_tx_tx_F": float("nan")}, "target C_tx_tx_F"),
        ({"keep_project": 0}, "retained eighth-symmetry Standard"),
        ({"N1": 5}, "differs from the authenticated source task stratum"),
        (
            {"physics_data_revision": "foreign-physics"},
            "differs from the base cohort",
        ),
    ],
)
def test_truth_row_fails_closed_on_physics_or_target_drift(
    tmp_path, updates, message
):
    truth = _truth(tmp_path, task_id=5000, n1=6)
    changed = dict(truth.result)
    changed.update(updates)
    truth = ingest.AuthenticatedTruth(
        **{**truth.__dict__, "result": changed}
    )

    with pytest.raises(ingest.StrictALIngestError, match=message):
        ingest._validate_truth_row(
            truth,
            profile=_profile(),
            physics_data_revision=PHYSICS_DATA_REVISION,
        )


def test_prepare_rejects_duplicate_base_task_identity(tmp_path, monkeypatch):
    base = _base_dataset(tmp_path)
    truth = _truth(tmp_path, task_id=100, n1=6)
    monkeypatch.setattr(
        ingest, "authenticate_collection", lambda _path: truth
    )

    with pytest.raises(
        ingest.StrictALIngestError, match="task_id already exists"
    ):
        ingest.prepare_ingest(
            base_dataset=base,
            expected_base_sha256=ingest._sha256_file(base),
            expected_base_rows=8,
            collection_paths=[truth.collection_path],
        )


def test_truth_row_rejects_non_n1_6_even_when_source_matches(tmp_path):
    truth = _truth(tmp_path, task_id=5001, n1=5)

    with pytest.raises(
        ingest.StrictALIngestError, match="not targeted N1=6"
    ):
        ingest._validate_truth_row(
            truth,
            profile=_profile(),
            physics_data_revision=PHYSICS_DATA_REVISION,
        )


def test_admission_requires_eight_unique_complete_geometries():
    facts = [
        {
            "source_task_payload_sha256": _sha(f"source-{index % 4}"),
            "candidate_physics_sha256": _sha(f"geometry-{index % 7}"),
            "N1": 6,
        }
        for index in range(8)
    ]

    admission = ingest._admission(
        facts,
        minimum_useful_rows=8,
        minimum_source_tasks=4,
    )

    assert admission["allowed"] is False
    assert admission["unique_complete_geometries"] == 7
    assert "unique_complete_geometries<8" in admission["reasons"]


def test_generic_json_collection_is_rejected_without_adapter(tmp_path):
    path = tmp_path / "raw-result.json"
    path.write_text(
        json.dumps({
            "schema_version": "generic-result-v1",
            "result": _strict_row(task_id=9000, n1=5),
        }),
        encoding="utf-8",
    )

    with pytest.raises(
        ingest.StrictALIngestError,
        match="raw JSON/CSV ingestion is forbidden",
    ):
        ingest.authenticate_collection(path)


def test_diagnostic_adapter_uses_finalized_loader_boundary(
    tmp_path, monkeypatch
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan_record = ingest._file_record(plan_path)
    collection_path = tmp_path / "diagnostic-collection.json"
    collection_path.write_text(
        json.dumps({
            "schema_version": ingest.DIAGNOSTIC_COLLECTION_SCHEMA,
            "plan": plan_record,
        }),
        encoding="utf-8",
    )
    result = _strict_row(task_id=9100, n1=6)
    plan = {
        "payload_sha256": _sha("diagnostic-plan"),
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
    }
    selected = {
        "task_identity": {
            "payload_sha256": _sha("diagnostic-source-task"),
            "seed": 9100,
            "fixed_primary_turns": 6,
        }
    }
    collection = {
        "schema_version": ingest.DIAGNOSTIC_COLLECTION_SCHEMA,
        "payload_sha256": _sha("diagnostic-collection"),
        "result": result,
        "result_sha256": canonical_sha256(result),
        "candidate_physics_sha256": _sha("diagnostic-candidate"),
        "task_id": 9100,
        "scheduler_status": "completed",
        "scheduler_mutation_performed": False,
        "selected_candidate_identity": {
            "candidate_physics_sha256": _sha("diagnostic-candidate"),
            "source_task_payload_sha256": _sha(
                "diagnostic-source-task"
            ),
            "source_result_sha256": _sha("diagnostic-source-result"),
            "seed": 9100,
            "fixed_primary_turns": 6,
        },
        "truth_evidence": {"authenticated": True},
        **ingest.DIAGNOSTIC_FLAGS,
    }
    calls = []

    def authenticate_collection(observed, *, predictor=None):
        calls.append(("collection", observed, predictor))
        return {
            "schema_version": (
                ingest.DIAGNOSTIC_AUTHENTICATED_COLLECTION_SCHEMA
            ),
            "collection": collection,
            "plan": plan,
            "params": {"params": True},
            "selected": selected,
            "submission": {"payload_sha256": _sha("submission")},
        }

    module = SimpleNamespace(
        COLLECTION_SCHEMA=ingest.DIAGNOSTIC_COLLECTION_SCHEMA,
        authenticate_collection=authenticate_collection,
    )
    monkeypatch.setattr(
        ingest.importlib,
        "import_module",
        lambda name: (
            module
            if name == "tools.mft_goal_diagnostic_standard_probe"
            else pytest.fail(f"unexpected module import: {name}")
        ),
    )

    truth = ingest.authenticate_collection(collection_path)

    assert truth.adapter_kind == "diagnostic"
    assert truth.source_task_payload_sha256 == _sha(
        "diagnostic-source-task"
    )
    assert calls == [("collection", collection_path.resolve(), None)]
