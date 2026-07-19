from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from tools.tier1_resonance_feedback import MANIFEST_SCHEMA, MODEL_SCHEMA, _json_sha
from tools.tier1_terminal_followup import TERMINAL_ANALYSIS_SCHEMA
from tools.tier1_thermal_acquisition import prepare_thermal_acquisition


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path]:
    model = root / "model"
    report_path = model / "training_report.json"
    _write(report_path, {
        "schema_version": MODEL_SCHEMA,
        "solver_revision": "a" * 40,
        "library_revision": "b" * 40,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    })
    manifest_path = model / "manifest.json"
    _write(manifest_path, {
        "schema_version": MANIFEST_SCHEMA,
        "production_eligible": False,
        "files": {"training_report.json": _sha(report_path)},
    })
    result_path = root / "result.json"
    _write(result_path, {"terminal": True})
    params = {
        "N1": 8, "N2": 80, "matrix_on": 1, "cap_on": 1,
        "loss_on": 0, "thermal_on": 0,
    }
    geometry = {
        "analytical_flux_density_limit": -0.1,
        "decoded_space_shrink": 0.0,
        "minimum_physical_insulation": 0.0,
        "core_group_manufacturability_limit": -1.0,
        "exterior_width_limit": 0.0,
        "exterior_length_limit": -1.0,
        "exterior_height_limit": 0.0,
    }
    rows = []
    for seed, thermal in ((10, 3.0), (11, 1.0)):
        candidate_params = {**params, "N1": seed}
        rows.append({
            "source_seed": seed,
            "decoded_params": candidate_params,
            "decoded_params_sha256": _json_sha(candidate_params),
            "constraint_G": {
                **geometry,
                "temperature_robust_limit:T_max_Tx": thermal,
                "temperature_robust_limit:T_max_core": -2.0,
            },
        })
    analysis_path = root / "terminal_analysis.json"
    _write(analysis_path, {
        "schema_version": TERMINAL_ANALYSIS_SCHEMA,
        "model_manifest_sha256": _sha(manifest_path),
        "result_references": [{
            "path": str(result_path), "sha256": _sha(result_path)
        }],
        "near_candidates": rows,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "fea_submission_approved": False,
    })
    return analysis_path, model


def test_thermal_lane_is_separate_nonexecuting_and_strict_full():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        analysis, model = _fixture(root)
        proposal = prepare_thermal_acquisition(
            terminal_analysis=analysis,
            model_dir=model,
            output=root / "output",
            candidate_limit=2,
        )
    assert proposal["selected_candidate_count"] == 2
    assert proposal["execution_authorized"] is False
    assert proposal["submission_performed"] is False
    assert proposal["production_eligible"] is False
    assert proposal["candidates"][0]["source_seed"] == 11
    for candidate in proposal["candidates"]:
        params = candidate["effective_params"]
        assert (params["matrix_on"], params["cap_on"]) == (1, 1)
        assert (params["loss_on"], params["thermal_on"]) == (1, 1)
        assert candidate["strict_full_result_contract_required"] is True


def test_thermal_lane_fails_on_terminal_result_tamper():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        analysis, model = _fixture(root)
        (root / "result.json").write_text("tampered", encoding="utf-8")
        try:
            prepare_thermal_acquisition(
                terminal_analysis=analysis,
                model_dir=model,
                output=root / "output",
            )
        except RuntimeError as error:
            assert "result reference SHA" in str(error)
        else:
            raise AssertionError("tampered terminal result must fail closed")
