import hashlib
import json
import os
from pathlib import Path

import pytest

from regression_260707.monitoring import deadline_design, readers
from regression_260707.monitoring.deadline_hit_design import _canonical_sha256


DEFAULT_SELECTOR = Path(
    r"\\RaiDrive-peets\ANSYS\git\MFT_1MW_2026"
    r"\artifacts\deadline_20260724\8010-full-em-hit-candidates"
    r"\first-complete-pass-selection.json"
)


def _selector() -> Path:
    configured = os.environ.get("MFT_DEADLINE_HIT_TEST_SELECTOR")
    path = Path(configured) if configured else DEFAULT_SELECTOR
    if not path.is_file():
        pytest.skip(f"deadline HIT integration selector is unavailable: {path}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_split_selector_is_fail_closed_for_solve():
    path = _selector()

    loaded = deadline_design.load_publication(path, _sha256(path))
    candidate = loaded["candidate"]

    assert candidate["validation_state"] == (
        "provisional_split_validated_complete_full_pending"
    )
    assert candidate["validation_badge"] == (
        "Full-EM + Standard PASS / complete Full pending"
    )
    assert candidate["cooling_variant"] in {
        "tim1k3fan6",
        "tim1k3fan7",
        "tim1k3fan8",
        "tim1k3fan9",
    }
    assert candidate["full_em_actual_pass"] is True
    assert candidate["standard_corroboration_pass"] is True
    assert candidate["complete_full_pending"] is True
    assert candidate["final_design_approved"] is False
    assert candidate["gui_launch_eligible"] is True
    assert candidate["gui_build_eligible"] is True
    assert candidate["gui_solve_eligible"] is False
    assert candidate["pred_f_res_min_screen_Hz"] >= 15_000.0
    assert candidate["size_W_mm"] <= 1_200.0
    assert candidate["size_L_mm"] <= 1_200.0
    assert candidate["size_H_mm"] <= 750.0


def test_resealed_mutating_selector_authority_is_rejected(tmp_path):
    source = _selector()
    selector = json.loads(source.read_text(encoding="utf-8"))
    selector["authority"]["live_8010_mutated"] = True

    selector_body = dict(selector)
    selector_body.pop("payload_sha256", None)
    selector["payload_sha256"] = _canonical_sha256(selector_body)
    path = tmp_path / "mutating-selector.json"
    path.write_text(
        json.dumps(selector, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="authority drifted",
    ):
        deadline_design.load_publication(path, _sha256(path))


def test_actual_hit_generation_detail_is_selectable(
    tmp_path, monkeypatch
):
    path = _selector()
    sha256 = _sha256(path)
    monkeypatch.setenv(deadline_design.PATH_ENV, str(path))
    monkeypatch.setenv(deadline_design.SHA256_ENV, sha256)

    class Scheduler:
        base_url = "http://127.0.0.1:8002"

        @staticmethod
        def mft_pipeline_status():
            return {}

    service = readers.ArtifactService(
        tmp_path / "regression",
        scheduler=Scheduler(),
        record_runtime=False,
    )
    snapshot = service._continuous_nsga2()
    generation_id = snapshot["selected_generation_id"]

    assert generation_id.startswith("tier1-hit-")
    assert snapshot["candidate_count"] == 1
    assert snapshot["display_candidate_count"] == 1
    assert snapshot["valid_candidate_count"] == 1
    assert snapshot["summary"]["valid_candidate_count"] == 1
    assert len(snapshot["candidates"]) == 1
    assert snapshot["candidates"][0]["validation_state"] == (
        "provisional_split_validated_complete_full_pending"
    )
    assert snapshot["candidates"][0]["final_design_approved"] is False
    assert snapshot["candidates"][0]["gui_solve_eligible"] is False
    detail = service.nsga2_generation(generation_id)
    assert detail["available"] is True
    assert detail["integrity_verified"] is True
    assert len(detail["candidates"]) == 1
    assert detail["candidates"][0]["gui_build_eligible"] is True
    assert detail["candidates"][0]["gui_solve_eligible"] is False
