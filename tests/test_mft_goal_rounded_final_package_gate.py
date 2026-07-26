from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import pytest

from tools import mft_goal_official5_rounded_full_prepare as full_prepare
from tools import mft_goal_rounded_final_package_gate as gate


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _inventory(
    source: Path,
    package_path: str,
    role: str,
    *,
    scientific: bool,
) -> dict[str, Any]:
    return gate._source_inventory_record(  # noqa: SLF001
        source,
        package_path=package_path,
        role=role,
        scientific_authority=scientific,
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    result = {
        "standard_plan": _write(tmp_path / "standard-plan.json", b"{}"),
        "standard_final": _write(tmp_path / "standard-final.json", b"{}"),
        "standard_collection": tmp_path / "standard-collection",
        "symmetric_fallback": _write(
            tmp_path / "fallback-symmetric.aedt", b"diagnostic symmetric"
        ),
        "full_terminal": tmp_path / "full-terminal",
        "full_checkpoint": tmp_path / "full-checkpoint",
        "aggregate": tmp_path / "aggregate",
        "html": _write(tmp_path / "pareto.html", b"<html>pareto</html>"),
        "pptx": _write(tmp_path / "drawing.pptx", b"PK\x03\x04pptx"),
        "pdf": _write(tmp_path / "drawing.pdf", b"%PDF-1.7\npdf"),
    }
    for key in (
        "standard_collection",
        "full_terminal",
        "full_checkpoint",
        "aggregate",
    ):
        result[key].mkdir()
    return result


def _install_fake_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    paths = _paths(tmp_path)
    candidate_params = _write(tmp_path / "candidate.json", b'{"N1":6}')
    symmetric_model = _write(
        tmp_path / "scientific-symmetric.aedt", b"symmetric model"
    )
    symmetric_result = _write(
        tmp_path / "scientific-symmetric-result.json", b'{"pass":true}'
    )
    full_model = _write(tmp_path / "terminal-full.aedt", b"full model")
    full_result = _write(
        tmp_path / "terminal-full-result.json", b'{"pass":true}'
    )
    checkpoint_model = _write(
        tmp_path / "checkpoint-full.aedt", b"checkpoint model"
    )
    pareto_csv = _write(tmp_path / "pareto.csv", b"x,y\n1,2\n")

    def candidate(_plan: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {
                "candidate_physics_sha256": gate.CANDIDATE_PHYSICS_SHA256,
                "selection_order": 5,
                "rounding_policy": copy.deepcopy(gate.ROUNDING_POLICY),
                "fixed_boundary": copy.deepcopy(gate.FIXED_BOUNDARY),
            },
            [
                _inventory(
                    candidate_params,
                    "design/candidate5_rounded_params.json",
                    "candidate_parameters",
                    scientific=True,
                )
            ],
        )

    def symmetric(
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {
                "mode": "task96340_terminal_scientific_collection",
                "source_task_id": gate.STANDARD_TASK_ID,
                "candidate_physics_sha256": gate.CANDIDATE_PHYSICS_SHA256,
                "scientific_pass": True,
                "model_delivery_allowed": True,
            },
            [
                _inventory(
                    symmetric_model,
                    "models/symmetric.aedt",
                    "symmetric_model",
                    scientific=True,
                ),
                _inventory(
                    symmetric_result,
                    "results/symmetric/result.json",
                    "symmetric_result",
                    scientific=True,
                ),
            ],
        )

    def symmetric_fallback(
        path: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {
                "mode": "diagnostic_model_file_fallback",
                "source_task_id": None,
                "candidate_physics_sha256": gate.CANDIDATE_PHYSICS_SHA256,
                "scientific_pass": False,
                "model_delivery_allowed": True,
                "diagnostic_file_fallback": True,
            },
            [
                _inventory(
                    path,
                    "models/symmetric.aedt",
                    "symmetric_fallback",
                    scientific=False,
                )
            ],
        )

    def full_terminal(
        _root: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {
                "mode": "terminal_full_collection",
                "candidate_physics_sha256": gate.CANDIDATE_PHYSICS_SHA256,
                "scientific_pass": True,
                "crosscheck_complete": True,
                "crosscheck_passed": True,
                "model_delivery_allowed": True,
            },
            [
                _inventory(
                    full_model,
                    "models/full_model.aedt",
                    "full_model",
                    scientific=True,
                ),
                _inventory(
                    full_result,
                    "results/full/result.json",
                    "full_result",
                    scientific=True,
                ),
            ],
        )

    def full_checkpoint(
        _root: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {
                "mode": "diagnostic_full_geometry_setup_checkpoint",
                "candidate_physics_sha256": gate.CANDIDATE_PHYSICS_SHA256,
                "scientific_pass": False,
                "crosscheck_complete": False,
                "crosscheck_passed": False,
                "model_delivery_allowed": True,
                "diagnostic_only": True,
            },
            [
                _inventory(
                    checkpoint_model,
                    "models/full_model.aedt",
                    "full_checkpoint",
                    scientific=False,
                )
            ],
        )

    def pareto(
        _root: Path, _html: Path
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return (
            {
                "seed_count": 512,
                "all_seed_global_nondominated_sort": True,
                "candidate_5_search_identity": {
                    "candidate_physics_sha256": (
                        gate.CANDIDATE_PHYSICS_SHA256
                    )
                },
            },
            [
                _inventory(
                    pareto_csv,
                    "search/global_pareto_front.csv",
                    "pareto",
                    scientific=True,
                )
            ],
        )

    monkeypatch.setattr(gate, "_authenticate_candidate_inputs", candidate)
    monkeypatch.setattr(
        gate, "_authenticate_symmetric_scientific", symmetric
    )
    monkeypatch.setattr(
        gate, "_authenticate_symmetric_fallback", symmetric_fallback
    )
    monkeypatch.setattr(gate, "_authenticate_full_terminal", full_terminal)
    monkeypatch.setattr(
        gate, "_authenticate_full_checkpoint", full_checkpoint
    )
    monkeypatch.setattr(gate, "_authenticate_pareto", pareto)
    paths["scientific_symmetric_model"] = symmetric_model
    return paths


def _prepare(
    paths: dict[str, Path],
    output: Path,
    *,
    scientific_symmetric: bool,
    full_checkpoint: bool,
    drawings: bool,
) -> Path:
    return gate.prepare(
        output=output,
        standard_plan=paths["standard_plan"],
        standard_final=(
            paths["standard_final"] if scientific_symmetric else None
        ),
        standard_collection=(
            paths["standard_collection"] if scientific_symmetric else None
        ),
        symmetric_fallback_aedt=(
            None if scientific_symmetric else paths["symmetric_fallback"]
        ),
        full_terminal_collection=(
            None if full_checkpoint else paths["full_terminal"]
        ),
        full_checkpoint_collection=(
            paths["full_checkpoint"] if full_checkpoint else None
        ),
        aggregate_root=paths["aggregate"],
        pareto_html=paths["html"],
        drawing_pptx=paths["pptx"] if drawings else None,
        drawing_pdf=paths["pdf"] if drawings else None,
    )


def _load_plan(path: Path) -> dict[str, Any]:
    return gate._validate_seal(  # noqa: SLF001
        gate._read_json(path, "test plan"),  # noqa: SLF001
        gate.PREPARE_SCHEMA,
        label="test plan",
    )


def test_scientific_terminal_sources_and_drawings_are_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _install_fake_authorities(monkeypatch, tmp_path)
    plan_path = _prepare(
        paths,
        tmp_path / gate.PREPARE_NAME,
        scientific_symmetric=True,
        full_checkpoint=False,
        drawings=True,
    )
    plan = _load_plan(plan_path)

    assert plan["schema_version"] == gate.PREPARE_SCHEMA
    assert plan["schema_version"] != "mft-goal-final-solver-package-v1"
    assert plan["readiness"] == {
        "symmetric_scientific_pass": True,
        "full_scientific_pass": True,
        "full_crosscheck_complete": True,
        "full_crosscheck_passed": True,
        "model_file_delivery_allowed": True,
        "scientific_package_allowed": True,
        "drawings_complete": True,
        "final_package_publish_eligible": True,
        "pending_reasons": [],
    }
    assert plan["package_published"] is False
    assert plan["package_directory_created"] is False
    assert plan["legacy_task96324_96326_gate_used"] is False
    assert plan["package_target"]["atomic_publish_contract"][
        "single_os_replace_after_complete_validation"
    ] is True

    report_path = tmp_path / gate.VALIDATION_NAME
    report = gate.validate(
        plan_path=plan_path,
        report_path=report_path,
    )
    assert report["semantic_authorities_reauthenticated"] is True
    assert report["source_inventory_rehashed"] is True
    assert report["package_published"] is False
    assert report_path.is_file()


def test_symmetric_file_fallback_can_deliver_models_but_never_science(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _install_fake_authorities(monkeypatch, tmp_path)
    plan = _load_plan(
        _prepare(
            paths,
            tmp_path / gate.PREPARE_NAME,
            scientific_symmetric=False,
            full_checkpoint=False,
            drawings=True,
        )
    )

    assert plan["symmetric_authority"]["diagnostic_file_fallback"] is True
    assert plan["readiness"]["model_file_delivery_allowed"] is True
    assert plan["readiness"]["scientific_package_allowed"] is False
    assert plan["readiness"]["final_package_publish_eligible"] is False
    assert any(
        "symmetric scientific PASS" in reason
        for reason in plan["readiness"]["pending_reasons"]
    )


def test_full_checkpoint_is_model_only_and_crosscheck_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _install_fake_authorities(monkeypatch, tmp_path)
    plan = _load_plan(
        _prepare(
            paths,
            tmp_path / gate.PREPARE_NAME,
            scientific_symmetric=True,
            full_checkpoint=True,
            drawings=False,
        )
    )

    assert plan["full_authority"]["diagnostic_only"] is True
    assert plan["readiness"]["model_file_delivery_allowed"] is True
    assert plan["readiness"]["full_crosscheck_complete"] is False
    assert plan["readiness"]["scientific_package_allowed"] is False
    assert plan["readiness"]["drawings_complete"] is False
    assert any(
        "cross-check is incomplete" in reason
        for reason in plan["readiness"]["pending_reasons"]
    )


def test_validate_fails_after_any_inventoried_source_byte_drifts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _install_fake_authorities(monkeypatch, tmp_path)
    plan_path = _prepare(
        paths,
        tmp_path / gate.PREPARE_NAME,
        scientific_symmetric=True,
        full_checkpoint=False,
        drawings=True,
    )
    paths["scientific_symmetric_model"].write_bytes(b"drifted bytes")

    with pytest.raises(
        gate.RoundedPackageGateError,
        match="prepare semantics drifted",
    ):
        gate.validate(plan_path=plan_path)


def test_exactly_one_symmetric_and_full_source_is_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _install_fake_authorities(monkeypatch, tmp_path)
    common = {
        "standard_plan": paths["standard_plan"],
        "aggregate_root": paths["aggregate"],
        "pareto_html": paths["html"],
        "drawing_pptx": None,
        "drawing_pdf": None,
    }
    with pytest.raises(
        gate.RoundedPackageGateError, match="exactly one symmetric"
    ):
        gate._build_payload(  # noqa: SLF001
            **common,
            standard_final=paths["standard_final"],
            standard_collection=paths["standard_collection"],
            symmetric_fallback_aedt=paths["symmetric_fallback"],
            full_terminal_collection=paths["full_terminal"],
            full_checkpoint_collection=None,
        )
    with pytest.raises(
        gate.RoundedPackageGateError, match="exactly one Full"
    ):
        gate._build_payload(  # noqa: SLF001
            **common,
            standard_final=paths["standard_final"],
            standard_collection=paths["standard_collection"],
            symmetric_fallback_aedt=None,
            full_terminal_collection=paths["full_terminal"],
            full_checkpoint_collection=paths["full_checkpoint"],
        )


def _aggregate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "aggregate-real"
    root.mkdir()
    files = {
        "global_terminal_candidates": _write(
            root / "global_terminal_candidates.csv", b"a,b\n1,2\n"
        ),
        "global_pareto_front": _write(
            root / "global_pareto_front.csv", b"a,b\n"
        ),
        "global_objective_front": _write(
            root / "global_objective_front.csv", b"a,b\n1,2\n"
        ),
        "standard_candidates": _write(
            root / "standard_candidates.csv",
            (
                "candidate_physics_sha,standard_selection_order,source_seed,"
                "source_task_id,standard_selection_roles,"
                "standard_selection_basis\n"
                f"{gate.CANDIDATE_PHYSICS_SHA256},5,2607262233,"
                "95913,objective_space_maximin,near_feasible_fallback\n"
            ).encode(),
        ),
    }
    artifact_names = {
        key: {
            "path": path.name,
            "sha256": gate._sha256_file(path),  # noqa: SLF001
        }
        for key, path in files.items()
    }
    manifest = gate._sealed(  # noqa: SLF001
        {
            "schema_version": gate.AGGREGATE_SCHEMA,
            "campaign_id": gate.CAMPAIGN_ID,
            "seed_count": gate.EXPECTED_SEED_COUNT,
            "minimum_seed_count": gate.EXPECTED_SEED_COUNT,
            "seeds": list(range(gate.EXPECTED_SEED_COUNT)),
            "input_terminal_row_count": gate.EXPECTED_TERMINAL_ROW_COUNT,
            "deduplicated_physical_geometry_count": (
                gate.EXPECTED_DEDUPLICATED_GEOMETRY_COUNT
            ),
            "global_objective_front_count": (
                gate.EXPECTED_OBJECTIVE_FRONT_COUNT
            ),
            "global_pareto_count": gate.EXPECTED_PRODUCTION_PARETO_COUNT,
            "standard_candidate_count": (
                gate.EXPECTED_STANDARD_CANDIDATE_COUNT
            ),
            "seed_local_pareto_merge_used": False,
            "sorting_authority": (
                "all_authenticated_terminal_rows_then_physical_dedupe_then_"
                "decoder_and_physical_G_and_surrogate_physicality_feasible_"
                "then_exact_2d_nlogn_non_dominated_sort"
            ),
            "artifacts": artifact_names,
        }
    )
    (root / "aggregate_manifest.json").write_bytes(
        gate._canonical_bytes(manifest) + b"\n"  # noqa: SLF001
    )
    html = _write(tmp_path / "global-pareto-audit.html", b"<html></html>")
    return root, html


def test_global_pareto_authority_requires_exact_all_seed_nds(
    tmp_path: Path,
) -> None:
    root, html = _aggregate_fixture(tmp_path)
    authority, inventory = gate._authenticate_pareto(  # noqa: SLF001
        root, html
    )

    assert authority["seed_count"] == 512
    assert authority["all_seed_global_nondominated_sort"] is True
    assert authority["seed_local_pareto_merge_used"] is False
    assert authority["candidate_5_search_identity"]["selection_order"] == 5
    assert {
        item["package_path"] for item in inventory
    } >= {
        "search/global_terminal_candidates.csv",
        "search/global_pareto_front.csv",
        "search/global_objective_front.csv",
        "search/global-pareto-audit.html",
    }

    (root / "global_terminal_candidates.csv").write_bytes(b"drift")
    with pytest.raises(
        gate.RoundedPackageGateError,
        match="global_terminal_candidates artifact bytes drifted",
    ):
        gate._authenticate_pareto(root, html)  # noqa: SLF001


def _relative_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": gate._sha256_file(path),  # noqa: SLF001
        "size_bytes": path.stat().st_size,
    }


def test_checkpoint_collection_is_explicitly_non_scientific(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint-collection"
    root.mkdir()
    model = _write(
        root / "full_model_geometry_setup_checkpoint.aedt",
        b"rounded setup-only AEDT",
    )
    receipt = full_prepare.sealed(
        {
            "schema_version": (
                gate.full_collector.CHECKPOINT_COLLECTION_SCHEMA
            ),
            "task_id": 97_000,
            "source_standard_task_id": gate.STANDARD_TASK_ID,
            "source_candidate_physics_sha256": (
                gate.CANDIDATE_PHYSICS_SHA256
            ),
            "solver_revision": "a" * 40,
            "library_revision": "b" * 40,
            "rounded_identity": copy.deepcopy(gate.ROUNDING_POLICY),
            "fixed_boundary": copy.deepcopy(gate.FIXED_BOUNDARY),
            "retained_checkpoint_aedt": _relative_record(model),
            "scientific_result_available": False,
            "scientific_pass": False,
            "thermal_pass": False,
            "production_package_eligible": False,
            "diagnostic_only": True,
        }
    )
    receipt_path = root / "collection_receipt.json"
    receipt_path.write_bytes(full_prepare.canonical_bytes(receipt) + b"\n")
    seal = full_prepare.sealed(
        {
            "schema_version": (
                gate.full_collector.CHECKPOINT_COLLECTION_SEAL_SCHEMA
            ),
            "task_id": 97_000,
            "collection_receipt": _relative_record(receipt_path),
            "scientific_pass": False,
            "production_promotion_eligible": False,
        }
    )
    (root / "collection_seal.json").write_bytes(
        full_prepare.canonical_bytes(seal) + b"\n"
    )

    authority, inventory = gate._authenticate_full_checkpoint(  # noqa: SLF001
        root
    )
    assert authority["scientific_pass"] is False
    assert authority["crosscheck_complete"] is False
    assert authority["model_delivery_allowed"] is True
    assert authority["diagnostic_only"] is True
    assert next(
        item
        for item in inventory
        if item["package_path"] == "models/full_model.aedt"
    )["scientific_authority"] is False


def test_cli_exposes_no_publish_or_assemble_command() -> None:
    parser = gate._parser()  # noqa: SLF001
    subparsers: list[Callable[..., Any]] = [
        action
        for action in parser._actions  # noqa: SLF001
        if hasattr(action, "choices")
    ]
    choices = next(action.choices for action in subparsers if action.choices)
    assert set(choices) == {"prepare", "validate"}
