import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from module.core_material_contract import PHYSICS_DATA_REVISION
from regression_260707.model_targets import CORE_REGION_TEMPERATURE_TARGETS
from regression_260707.monitoring.app import create_app


class _ClientAddressOverride:
    """Set an ASGI client address across Starlette TestClient versions."""

    def __init__(self, app, client):
        self.app = app
        self.client = client

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            scope = {**scope, "client": self.client}
        await self.app(scope, receive, send)


def _addressed_client(app, *, base_url, client):
    return TestClient(
        _ClientAddressOverride(app, client),
        base_url=base_url,
    )


def test_operational_route_contract_is_preserved(artifact_service):
    schema = create_app(service=artifact_service).openapi()
    methods = {
        (method.upper(), path)
        for path, operations in schema["paths"].items()
        for method in operations
    }

    assert {
        ("GET", "/api/nsga2/generations/{generation_id}"),
        ("GET", "/api/nsga2/progress"),
        ("GET", "/api/local-aedt-gui/status"),
        ("GET", "/api/local-aedt-gui/launches"),
        ("POST", "/api/operator/local-aedt-gui/launch"),
        ("GET", "/api/compute-campaign"),
    } <= methods


def test_dashboard_page_and_all_read_only_apis(artifact_service):
    client = TestClient(create_app(service=artifact_service))
    page = client.get("/")
    assert page.status_code == 200
    assert '<meta name="color-scheme" content="light">' in page.text
    assert "/static/app.css?v=20260725-slurm-allocation-jobs-v2" in page.text
    assert "/static/app.js?v=20260725-slurm-allocation-jobs-v2" in page.text
    assert 'id="codex-work-panel"' in page.text
    assert 'id="codex-campaign-strip"' in page.text
    assert "활성 Slurm allocation job" in page.text
    assert 'id="codex-campaign-allocation-jobs"' in page.text
    assert 'id="codex-campaign-submitted"' in page.text
    assert 'id="codex-campaign-running"' in page.text
    assert 'id="codex-campaign-collections"' in page.text
    assert 'id="codex-current-list"' in page.text
    assert 'id="codex-completed-list"' in page.text
    assert 'id="codex-attention-list"' in page.text
    assert 'id="model-parity-provenance"' in page.text
    assert "최적설계 파이프라인" in page.text
    assert "data-chart" in page.text
    assert 'id="nsga-tier1-progress"' in page.text
    assert 'id="nsga-generation-selector"' in page.text
    assert 'role="tablist"' in page.text
    assert 'id="nsga-generation-state"' in page.text
    assert 'id="nsga-generation-summary"' in page.text
    assert 'id="sealed-successor-panel"' in page.text
    assert 'id="sealed-successor-candidates"' in page.text
    assert 'id="sealed-successor-detail-button"' in page.text
    assert 'id="sealed-successor-dialog"' in page.text
    assert 'id="sealed-successor-validation"' in page.text
    assert 'id="sealed-successor-standard-fea"' in page.text
    assert 'id="sealed-successor-standard-fea-temperatures"' in page.text
    assert "Sealed successor / FEA pending" in page.text
    assert "canonical Pareto 후보나 terminal FEA 결과가 아닙니다" in page.text
    assert "코호트 상세 →" in page.text
    assert 'href="/cohorts"' in page.text
    assert 'id="data-member-shas"' in page.text
    assert "전체 수집 Raw" in page.text
    assert "data-raw-total" in page.text
    assert "최근 시뮬레이션 단계별 소요시간" in page.text
    assert "stage-time-matrix-mean" in page.text
    assert "stage-time-electrostatic-mean" in page.text
    assert "Electrostatic 평균" in page.text
    assert 'id="stage-timing-basis"' in page.text
    assert "활성 코호트 기준" in page.text
    assert "활성 코호트 확인 중" in page.text
    assert "활성 코호트 타이밍 데이터 없음" in page.text
    assert "단계 소요시간" in page.text
    assert "final-time-matrix" in page.text
    assert "기존 MFT 프로젝트 병렬 정책" in page.text
    assert "현재 512-seed Slurm 캠페인의 제출·실행 수가 아닙니다" in page.text
    assert "95684–96195" in page.text
    assert 'id="verify-deadline-campaign"' in page.text
    assert 'id="verify-campaign-summary"' in page.text
    assert "기존 pipeline Standard FEA 후보" in page.text
    assert 'id="parallel-target-form" class="parallel-target-form"' in page.text
    assert "parallel-target-input" in page.text
    assert "parallel-effective-target" in page.text
    assert "parallel-validated-limit" in page.text
    assert "parallel-active" in page.text
    assert "parallel-solving" in page.text
    assert 'id="refill-controller-status" class="parallel-target-form"' in page.text
    assert 'id="parallel-target-status" class="parallel-target-status"' in page.text
    assert 'id="parallel-control-note" class="parallel-control-note"' in page.text
    assert "세대 ID 축약" in page.text
    assert "parallel-logical-active" in page.text
    assert "parallel-attaching" in page.text
    assert 'id="model-history-metric"' in page.text
    assert 'value="mape_pct"' in page.text
    assert 'id="cohort-list"' not in page.text
    assert 'id="cohort-history"' not in page.text
    assert 'id="cohort-lamination-factor"' in page.text
    assert 'id="cohort-flux-availability"' in page.text
    current_details_tag = '<details id="quarantine-current-details" class="quarantine-reason-details">'
    assert current_details_tag in page.text
    assert 'id="quarantine-current-reason-count"' in page.text
    assert '<span class="fold-open">펼치기 ▾</span>' in page.text
    assert '<span class="fold-close">접기 ▴</span>' in page.text
    assert 'id="quarantine-current-reasons"' in page.text
    current_details_start = page.text.index(current_details_tag)
    current_details_end = page.text.index("</details>", current_details_start)
    current_reasons_start = page.text.index('id="quarantine-current-reasons"')
    assert current_details_start < current_reasons_start < current_details_end
    assert 'id="quarantine-legacy-reasons"' in page.text
    assert 'id="capacitance-summary"' in page.text
    assert 'id="resonance-summary"' in page.text
    assert 'id="thermal-model-list"' in page.text
    assert 'id="thermal-model-basis"' in page.text
    assert 'id="aedt-attach-card"' in page.text
    assert 'id="aedt-license-usage"' in page.text
    assert 'id="aedt-pool-idle"' in page.text
    assert "중앙 풀(비활성·구조적 차단)" in page.text
    assert 'id="aedt-node-local-progress"' in page.text
    assert 'id="nsga-constraint-contract"' in page.text
    assert 'id="nsga-near-panel"' in page.text
    assert 'id="nsga-near-table"' in page.text
    assert "Near feasible · 제약 미통과 후보" in page.text
    assert 'id="local-aedt-result"' in page.text
    assert 'id="local-aedt-result-json"' in page.text
    assert 'id="data-active-training"' in page.text
    assert 'id="data-completed-training"' in page.text
    assert 'id="data-active-model"' in page.text
    assert 'id="pipeline-hpo-state"' in page.text
    assert 'id="compute-campaign-state"' in page.text
    assert 'id="compute-resource-cpus"' in page.text
    assert "Current7 controller 전환 (모델 독립)" in page.text
    assert "정책 target" in page.text
    assert "target CPU 상한" in page.text
    assert "known-node unplaced 상한" in page.text
    assert 'id="compute-resource-unplaced"' in page.text
    assert "HPO → 최종 학습 → 새 모델 탐색" in page.text
    assert "학습 데이터 수 기준" in page.text

    cohorts_page = client.get("/cohorts")
    assert cohorts_page.status_code == 200
    assert "수집 데이터 코호트" in cohorts_page.text
    assert "코호트 상세" in cohorts_page.text
    assert 'href="/"' in cohorts_page.text
    assert '<th>SHA</th><th>revision</th><th>Raw</th>' in cohorts_page.text
    assert '<th>Strict EM</th><th>Strict full</th><th>+/h</th>' in cohorts_page.text
    assert 'id="cohorts-body"' in cohorts_page.text
    assert '<meta name="color-scheme" content="light">' in cohorts_page.text
    assert "/static/app.css?v=20260723-pareto-light-v1" in cohorts_page.text
    assert "/static/cohorts.js?v=20260723-pareto-light-v1" in cohorts_page.text

    script = client.get("/static/app.js")
    assert "NOT THE DEADLINE CAMPAIGN" in script.text
    assert "현재 512-seed Slurm 캠페인의 job 수가 아닙니다" in script.text
    assert script.status_code == 200
    assert "function duration(value)" in script.text
    assert "function renderComputeCampaign(payload = {})" in script.text
    assert 'value.includes("failed")' in script.text
    assert "resources.policy_active_target" in script.text
    assert "resources.global_unplaced_limit" in script.text
    assert "`target ${number(policyTarget)}`" in script.text
    assert "data.current_physics_data_revision" in script.text
    assert "data.member_git_hash_shorts" in script.text
    assert "data.revision_raw_rows" in script.text
    assert "data.raw_total_rows" in script.text
    assert "격리" in script.text
    assert "data.simulation_timing" in script.text
    assert "timing.cohort_label" in script.text
    assert "timing.active_cohort" in script.text
    assert "data.active_cohort" in script.text
    assert "n=${number(timingWindowRows)}" in script.text
    assert "timingCell(evaluation.timing_seconds)" in script.text
    assert 'return "—"' in script.text
    assert 'fetch("/api/operator/simulation-policy"' in script.text
    assert '"X-MFT-Operator-Control": "simulation-policy-v1"' in script.text
    assert 'expected_revision: scheduler.policy_revision' in script.text
    assert 'scale_down_mode: "drain"' in script.text
    assert 'key === "no_refill_needed"' in script.text
    assert 'key === "pooled_bundle_pending"' in script.text
    assert 'key === "failed_closed"' in script.text
    assert "정상 (보충 불필요)" in script.text
    assert "보충 실행" in script.text
    assert "AEDT 공유 번들 진행 중" in script.text
    assert "오류로 안전정지 (관리자 확인 필요)" in script.text
    assert "컨트롤러 상태 파일을 찾을 수 없음" in script.text
    assert "?? refillController.concurrency_target" in script.text
    assert "scheduler.live_queued" in script.text
    assert "x: (item) => Number(item.n)" in script.text
    assert "historyPointTooltip" in script.text
    assert "CV P90 APE" in script.text
    assert "function renderParityProvenance(model, payload = {})" in script.text
    assert "function paritySampleR2(points)" in script.text
    assert "legacy quantized labels / 교정 재학습 대기" in script.text
    assert "unique actual ${number(provenance.unique_actual_count)}개" in script.text
    assert '"SHA 인증 교정 OOF overlay"' in script.text
    assert '`source ${provenance.overlay_path || "—"}`' in script.text
    assert '`sha256 ${provenance.overlay_sha256 || "—"}`' in script.text
    assert 'model.parity_provenance.source_kind === "authenticated_overlay"' in script.text
    assert 'const r2Label = isOverlay ? "표시 표본 R²" : "R²"' in script.text
    assert "모델 승격 전 SHA 인증 교정 OOF overlay" in script.text
    assert 'OOF checkpoint ${model.parity_checkpoint ?? "—"}' in script.text
    assert "data.current_cohort_metadata" in script.text
    assert "data.quarantine" in script.text
    assert '`${current.label || "활성 코호트"} — ${count(current.rows)}`' in script.text
    assert 'hasCurrentReasons ? count(currentReasons.length, "건") : "—"' in script.text
    assert "#quarantine-current-details" not in script.text
    assert "electrostatic.cap_stage_present_rows" in script.text
    assert '"C_tx_tx"' in script.text
    assert "resonance.interwinding" in script.text
    assert "resonance_minimum_required_Hz" in script.text
    assert "resonance_margin_Hz" in script.text
    assert "data.thermal_models" in script.text
    assert "scheduler.aedt_attach" in script.text
    assert "license.used" in script.text
    assert "pool.min_idle_sessions" in script.text
    assert "ratio(pool.hard_sessions, pool.max_sessions)" in script.text
    assert "ratio(pool.ready_sessions, pool.busy_sessions)" in script.text
    assert "attach.node_local" in script.text
    assert "nodeLocal.active_host_tasks" in script.text
    assert "노드 로컬: 활성 호스트" in script.text
    assert '["matrix", "loss", "electrostatic", "icepak", "total"]' in script.text
    assert "function renderLocalGuiResult(launch)" in script.text
    assert "completed_detached" in script.text
    assert "progress.terminal_results_verified" in script.text
    assert "progress.candidate_preview_truncated === true" in script.text
    assert "progress.candidate_preview_count" in script.text
    assert '"/api/nsga2/progress",' in script.text
    assert '"/api/local-aedt-gui/launches",' in script.text
    assert '"/api/dashboard",' in script.text
    assert '"/api/codex-work",' in script.text
    assert "function renderCodexWork(payload = {})" in script.text
    assert "function renderDeadlineCampaign(campaign = {})" in script.text
    assert "function deadlineCampaignVerified(campaign = {})" in script.text
    assert "완료·회수 0은 제출 0이 아닙니다" in script.text
    assert "job 제출 0건이라는 뜻이 아닙니다" in script.text
    assert "function refreshCodexWork(options)" in script.text
    assert "function refreshNsgaProgress(options)" in script.text
    assert "function refreshLocalGuiLaunches(options)" in script.text
    assert "function fetchJsonWithTimeout(url, options, timeoutMs)" in script.text
    assert "refreshNsgaProgress(options);" in script.text
    assert "refreshLocalGuiLaunches(options);" in script.text
    assert "await Promise.all([" not in script.text
    assert "progressPayload?.integrity_verified === true" in script.text
    assert "function renderSealedSuccessors(collection = {}, fallback = {})" in script.text
    assert "rootPayload.sealed_successors || {}" in script.text
    assert "rootPayload.sealed_successor || {}" in script.text
    assert "selectedNsgaGenerationId" in script.text
    assert "function nsgaGenerationSelection(payload)" in script.text
    assert "function renderNsgaGenerationSelector(rootPayload, selection)" in script.text
    assert "function tier1SourceProgress(search = {})" in script.text
    assert "function renderNsgaNearPreview(payload = {}, tier1 = {})" in script.text
    assert "function nearViolationLabel(violation = {})" in script.text
    assert "item.valid_pareto === false" in script.text
    assert "GUI/FEA 자동 제출 불가" in script.text
    assert "function renderScatter(candidates, nearCandidates = [])" in script.text
    assert 'chart_kind: "pareto"' in script.text
    assert 'chart_kind: "near"' in script.text
    assert 'class: "pareto-front-line"' in script.text
    assert 'const isNear = candidate.chart_kind === "near"' in script.text
    assert "근접 후보 ${number(nearTotal)}개 (주황색)" in script.text
    assert 'row.addEventListener("keydown", (event) =>' in script.text
    assert "generation.source_count" in script.text
    assert "generation.refused_terminal_count" in script.text
    assert "refusalSuffix" in script.text
    assert "candidate.tier1_source_cohort_id" in script.text
    assert "button.disabled = generation.selectable === false" in script.text
    assert "generation.selectable === false" in script.text
    assert "state.nsgaGenerationCache" in script.text
    assert "state.nsgaGenerationRequests" in script.text
    assert "function nsgaGenerationCacheKey(generation)" in script.text
    assert "function nsgaGenerationDetailVerified(detail)" in script.text
    assert "function beginNsgaGenerationRequest(cache, requests, generation)" in script.text
    assert "function settleNsgaGenerationRequest(cache, requests, request, detail)" in script.text
    assert "detail?.available === true && detail?.integrity_verified === true" in script.text
    assert "state.nsgaGenerationRequests.has(String(generation.id))" in script.text
    assert 'generation?.updated_at || ""' in script.text
    assert "`/api/nsga2/generations/${encodeURIComponent(generation.id)}`" in script.text
    assert "candidate.gui_launch_eligible === false" in script.text
    assert "이전 조건 세대는 비교용 읽기 전용" in script.text
    assert '"무결성 인증 실패"' in script.text
    assert '"조건 없음"' in script.text
    assert "selection.view?.integrity_verified === true" in script.text
    assert 'limit == null || limit === ""' in script.text
    assert 'value != null && value !== ""' in script.text
    assert 'viewingActiveGeneration ? "Active" : "Archived"' in script.text
    assert "function constraintCheckLabel(key, check = {})" in script.text
    assert "const limit = check.limit" in script.text
    assert 'candidate?.terminal_result === false' in script.text
    assert 'candidate?.canonical_candidate === false' in script.text
    assert 'candidate?.pareto === false' in script.text
    assert 'candidate?.production === false' in script.text
    assert 'candidate?.fea_submission_performed === false' in script.text
    assert "function populateSealedSuccessorDialog(payload)" in script.text
    assert "candidate.standard_fea_validation || {}" in script.text
    assert '"#sealed-successor-validation"' in script.text
    assert '"#sealed-successor-standard-fea"' in script.text
    assert '"#sealed-successor-standard-fea-temperatures"' in script.text
    assert "standard-fea-overlay" in script.text
    assert "canonical Pareto/terminal 개수에는 포함하지 않으며" in script.text
    assert '$("#sealed-successor-detail-button").addEventListener' in script.text
    assert "data.active_training_job" in script.text
    assert "pipeline.experimental_shadow_training" in script.text

    cohorts_script = client.get("/static/cohorts.js")
    assert cohorts_script.status_code == 200
    assert "function compactCohorts(payload)" in cohorts_script.text
    assert "legacy_aggregate: true" in cohorts_script.text
    assert "레거시 (${number(cohort.cohort_count)}개 코호트)" in cohorts_script.text
    assert 'fetch("/api/data"' in cohorts_script.text

    stylesheet = client.get("/static/app.css")
    assert ".sealed-successor-panel" in stylesheet.text
    assert ".codex-work-panel" in stylesheet.text
    assert ".codex-campaign-strip" in stylesheet.text
    assert ".verify-deadline-campaign" in stylesheet.text
    assert ".codex-work-column.attention" in stylesheet.text
    assert ".nsga-generation-selector" in stylesheet.text
    assert ".nsga-generation-button.active" in stylesheet.text
    assert ".nsga-generation-button:disabled" in stylesheet.text
    assert ".sealed-successor-candidate-grid" in stylesheet.text
    assert ".sealed-successor-candidate-card.active" in stylesheet.text
    assert ".standard-fea-overlay.running" in stylesheet.text
    assert stylesheet.status_code == 200
    assert ".timing-grid" in stylesheet.text
    assert ".stage-timing-grid" in stylesheet.text
    assert ".stage-timing-empty" in stylesheet.text
    assert ".history-metric-control" in stylesheet.text
    assert ".parity-provenance-note.invalid" in stylesheet.text
    assert ".parity-provenance-badge.valid" in stylesheet.text
    assert ".state-chip.parity_overlay" in stylesheet.text
    assert ".chart-tooltip" in stylesheet.text
    assert ".cohort-detail-table tr.active" in stylesheet.text
    assert ".cohort-detail-table tr.legacy-aggregate" in stylesheet.text
    assert ".quarantine-reason-details[open] .fold-open" in stylesheet.text
    assert ".quarantine-reason-details[open] .fold-close" in stylesheet.text
    assert ".quarantine-legacy" in stylesheet.text
    assert ".electrostatic-presence-grid" in stylesheet.text
    assert ".thermal-model-row" in stylesheet.text
    assert ".aedt-attach-card" in stylesheet.text
    assert ".aedt-node-local-progress" in stylesheet.text
    assert "color-scheme: light" in stylesheet.text
    assert "--bg: #f6f7f9" in stylesheet.text
    assert "--panel: #ffffff" in stylesheet.text
    assert "--accent: #1d6fb8" in stylesheet.text
    assert ".chart .pareto-front-line" in stylesheet.text
    assert ".chart .point.near" in stylesheet.text
    assert "fill: #d06a20" in stylesheet.text

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["data"]["total_rows"] == 1
    assert dashboard.json()["data"]["raw_total_rows"] == 2
    assert dashboard.json()["compute_campaign"]["read_only"] is True
    assert dashboard.json()["data"]["count_basis"] == "physics_revision_strict_full"
    assert dashboard.json()["data"]["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert dashboard.json()["data"]["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    data = dashboard.json()["data"]
    active = data["active_cohort"]
    assert active["available"] is True
    assert active["status"] == "active"
    assert active["git_hash"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert active["expected_physics_data_revision"] == PHYSICS_DATA_REVISION
    assert data["revision_raw_rows"] == 2
    assert data["member_git_hash_shorts"] == ["754923c", "bbbbbbb"]
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["active_cohort"] == active
    assert timing["cohort_rows"] == 1
    assert timing["window_rows"] == 1
    assert timing["stages"]["total"]["mean_seconds"] == 3000.0
    assert timing["stages"]["electrostatic"]["sample_count"] == 0

    assert client.get("/api/status").status_code == 200
    codex_work = client.get("/api/codex-work")
    assert codex_work.status_code == 200
    assert codex_work.json()["schema_version"] == "mft-codex-work-status-v1"
    assert codex_work.json()["available"] is False
    assert client.get("/api/data").json()["complete_rows"] == 1
    models = client.get("/api/models").json()
    assert models["trained_count"] == 2
    model_lookup = {item["target"]: item for item in models["models"]}
    expected_core_targets = (
        "Tprobe_core_center_max",
        *CORE_REGION_TEMPERATURE_TARGETS,
    )
    assert all(target in model_lookup for target in expected_core_targets)
    assert all(model_lookup[target]["status"] == "not_trained"
               for target in expected_core_targets)
    assert client.get("/api/models/Llt_phys/history").json()["target"] == "Llt_phys"
    parity = client.get("/api/models/Llt_phys/parity")
    assert parity.status_code == 200
    assert parity.json()["target"] == "Llt_phys"
    assert parity.json()["available"] is False  # active registry takes priority
    for target in ("B_max_core", "C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F"):
        parity = client.get(f"/api/models/{target}/parity")
        assert parity.status_code == 200
        assert parity.json()["target"] == target
    for target in expected_core_targets:
        history = client.get(f"/api/models/{target}/history")
        assert history.status_code == 200
        assert history.json()["target"] == target
    assert client.get("/api/models/not-a-target/history").status_code == 404
    assert client.get("/api/models/not-a-target/parity").status_code == 404
    assert client.get("/api/nsga2").json()["candidate_count"] == 2
    verification = client.get("/api/verification").json()
    assert verification["final"]["status"] == "pass"
    assert verification["standard_candidates"][0]["evaluation"]["timing_seconds"] == {
        "matrix": 353.31,
        "loss": 1720.78,
        "icepak": 1039.83,
        "total": 3113.92,
    }
    assert client.get("/api/history").status_code == 200
    assert client.get("/api/compute-campaign").json()["read_only"] is True
    assert client.get("/healthz").json()["status"] == "ok"
    for endpoint in (
        "/api/data",
        "/api/nsga2",
        "/api/nsga2/progress",
        "/api/compute-campaign",
    ):
        assert client.get(endpoint).headers["content-type"] == (
            "application/json; charset=utf-8"
        )


def test_nsga_generation_api_loads_history_without_joining_local_validation(
        tmp_path):
    calls = []
    validation_index_calls = []

    class StubService:
        def nsga2_generation(self, generation_id):
            calls.append(generation_id)
            return {
                "available": True,
                "selected_generation_id": generation_id,
                "generation": {
                    "id": generation_id,
                    "active": False,
                    "state": "archived",
                },
                "candidate_count": 1,
                "candidates": [{
                    "id": "archived-candidate-1",
                    "gui_launch_eligible": False,
                }],
            }

    class StubLauncher:
        def validation_index(self):
            validation_index_calls.append(True)
            return {
                "archived-candidate-1": {
                    "launch_id": "historical-validation-1",
                    "state": "completed_held",
                }
            }

    client = TestClient(create_app(
        regression_root=tmp_path,
        service=StubService(),
        local_gui_launcher=StubLauncher(),
    ))
    response = client.get("/api/nsga2/generations/tier1-history-v2")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    payload = response.json()
    assert calls == ["tier1-history-v2"]
    assert payload["selected_generation_id"] == "tier1-history-v2"
    assert payload["generation"]["state"] == "archived"
    assert payload["candidates"][0]["gui_launch_eligible"] is False
    assert payload["candidates"][0]["local_aedt_validation"] is None
    assert payload["local_aedt_validation_count"] == 0
    assert validation_index_calls == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_deadline_campaign_ui_contract_is_strict_and_distinguishes_zero_collection():
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    assert "Slurm allocation job ${number(allocationJobs)}" in app_js.read_text(
        encoding="utf-8"
    )
    script = r"""
const path = require("node:path");
const { deadlineCampaignVerified } = require(path.resolve(process.argv[1]));
const valid = {
  available: true,
  integrity_verified: true,
  allocation_jobs_active: 6,
  submitted_total: 59,
  running: 30,
  collections: 0,
  collection_zero_means_submission_zero: false,
};
process.stdout.write(JSON.stringify({
  valid: deadlineCampaignVerified(valid),
  missing: deadlineCampaignVerified({
    available: true,
    integrity_verified: true,
    collection_zero_means_submission_zero: false,
  }),
  inconsistent: deadlineCampaignVerified({
    ...valid,
    submitted_total: 2,
    running: 3,
  }),
  noAllocation: deadlineCampaignVerified({
    ...valid,
    allocation_jobs_active: 0,
  }),
  conflated: deadlineCampaignVerified({
    ...valid,
    collection_zero_means_submission_zero: true,
  }),
}));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(app_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )

    assert json.loads(completed.stdout) == {
        "valid": True,
        "missing": False,
        "inconsistent": False,
        "noAllocation": False,
        "conflated": False,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_cohorts_page_compacts_legacy_noise_and_keeps_active_first():
    cohorts_js = Path(__file__).resolve().parents[1] / "static" / "cohorts.js"
    script = r"""
const path = require("node:path");
const { compactCohorts } = require(path.resolve(process.argv[1]));
const revision = "mft1mw-1k101-native-lamination-kf0p85-v3";
const cohort = (sha, savedAt, raw = 1, physicsRevision = revision) => ({
  git_hash: sha.repeat(40), git_hash_short: sha.repeat(10),
  physics_data_revision: physicsRevision, latest_saved_at: savedAt,
  active: false, raw_rows: raw, strict_em_rows: 0,
  strict_full_rows: 0, growth_rate_per_hour: 0,
});
const cohorts = [
  { ...cohort("a", "2026-07-13T10:00:00+09:00", 8), active: true,
    strict_em_rows: 7, strict_full_rows: 6, growth_rate_per_hour: 2 },
  cohort("b", "2026-07-13T09:59:00.000900+09:00"),
  cohort("c", "2026-07-13T09:59:00.000700+09:00"),
  cohort("d", "2026-07-13T09:59:00.000600+09:00"),
  cohort("e", "2026-07-13T09:59:00.000800+09:00", 2, "legacy_unspecified"),
  cohort("f", "2026-07-13T09:59:00.000500+09:00", 3, "legacy_unspecified"),
];
process.stdout.write(JSON.stringify(compactCohorts(cohorts)));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(cohorts_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    rows = json.loads(completed.stdout)

    assert [row["git_hash_short"] for row in rows] == [
        "a" * 10,
        "b" * 10,
        "legacy",
        "c" * 10,
        "d" * 10,
    ]
    assert rows[0]["active"] is True
    aggregate = rows[2]
    assert aggregate["legacy_aggregate"] is True
    assert aggregate["cohort_count"] == 2
    assert aggregate["raw_rows"] == 5
    assert aggregate["latest_saved_at"] == "2026-07-13T09:59:00.000800+09:00"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_nsga_generation_detail_cache_retries_failures_and_deduplicates_inflight():
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = r"""
const path = require("node:path");
const {
  beginNsgaGenerationRequest,
  nsgaGenerationDetailVerified,
  settleNsgaGenerationRequest,
} = require(path.resolve(process.argv[1]));
const cache = new Map();
const requests = new Set();
const generation = {
  id: "archived-v2", cohort_id: "cohort-v2", updated_at: "2026-07-19T00:00:00Z",
  candidate_count: 3, display_candidate_count: 3, state: "archived",
};
const first = beginNsgaGenerationRequest(cache, requests, generation);
const duplicate = beginNsgaGenerationRequest(cache, requests, generation);
const unavailable = { available: false, integrity_verified: true };
const unavailableAccepted = settleNsgaGenerationRequest(
  cache, requests, first, unavailable,
);
const retry = beginNsgaGenerationRequest(cache, requests, generation);
const unverified = { available: true, integrity_verified: false };
const unverifiedAccepted = settleNsgaGenerationRequest(
  cache, requests, retry, unverified,
);
const immediateRetry = beginNsgaGenerationRequest(cache, requests, generation);
const verified = { available: true, integrity_verified: true, candidates: [] };
const verifiedAccepted = settleNsgaGenerationRequest(
  cache, requests, immediateRetry, verified,
);
const afterSuccess = beginNsgaGenerationRequest(cache, requests, generation);
process.stdout.write(JSON.stringify({
  first: Boolean(first), duplicate: Boolean(duplicate),
  unavailableVerified: nsgaGenerationDetailVerified(unavailable),
  unavailableAccepted, retry: Boolean(retry), unverifiedAccepted,
  immediateRetry: Boolean(immediateRetry), verifiedAccepted,
  afterSuccess: Boolean(afterSuccess), cacheSize: cache.size,
  requestsSize: requests.size,
}));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(app_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "first": True,
        "duplicate": False,
        "unavailableVerified": False,
        "unavailableAccepted": False,
        "retry": True,
        "unverifiedAccepted": False,
        "immediateRetry": True,
        "verifiedAccepted": True,
        "afterSuccess": False,
        "cacheSize": 1,
        "requestsSize": 0,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_nsga_progress_projection_requires_matching_authenticated_generation():
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = r"""
const path = require("node:path");
const {
  nsgaAuthoritySearch,
  nsgaAuthorityView,
  nsgaProgressMatchesDashboard,
} = require(path.resolve(process.argv[1]));
const source = (cohort) => ({
  role: "primary", label: "main", cohort_id: cohort,
  constraint_version: "res15k-t110", hard_spec_sha256: "a".repeat(64),
  model_manifest_sha256: "b".repeat(64),
  deployment_model_manifest_sha256: "c".repeat(64),
  nsga_code_revision: "d".repeat(40), accepted: true,
});
const search = {
  available: true, integrity_verified: true,
  constraint_version: "res15k-t110",
  model_manifest_sha256: "b".repeat(64),
  deployment_model_manifest_sha256: "c".repeat(64),
  source_count: 1, supplemental_source_count: 0,
  configured_source_count: 1, rejected_source_count: 0,
  sources: [source("cohort-15k")],
};
const progress = {
  ...search, counters_consistent: true,
};
const current7 = {
  available: true, integrity_verified: true,
  bundle_manifest_sha256: "e".repeat(64),
  index_file_sha256: "f".repeat(64),
  snapshot_file_sha256: "1".repeat(64),
  snapshot_sha256: "2".repeat(64),
  constraint_identity_sha256: "3".repeat(64),
  constraint_version: "current7-res15k-t110",
  bundle_id: "current7-bundle",
  cohort_id: "current7-cohort",
  constraints: { resonance_min_Hz: 15000 },
  search_count: 41,
  running_count: 24,
  queued_count: 5,
  attaching_count: 1,
  active_plus_queued: 30,
  completed_count: 8,
  failed_count: 3,
  cancelled_count: 0,
  timeout_count: 0,
  authenticated_terminal_seed_count: 7,
  refused_terminal_count: 1,
  feasible_pareto_count: 5,
};
const current7Progress = {
  available: true,
  integrity_verified: true,
  counters_consistent: true,
  coherent_snapshot: true,
  authority_kind: "current7",
  authority_identity_sha256: current7.bundle_manifest_sha256,
  authority_index_file_sha256: current7.index_file_sha256,
  authority_snapshot_sha256: current7.snapshot_file_sha256,
  authority_snapshot_identity_sha256: current7.snapshot_sha256,
  constraint_identity_sha256: current7.constraint_identity_sha256,
  constraint_version: current7.constraint_version,
  search_count: current7.search_count,
  running_count: current7.running_count,
  queued_count: current7.queued_count,
  attaching_count: current7.attaching_count,
  active_plus_queued: current7.active_plus_queued,
  completed_count: current7.completed_count,
  failed_count: current7.failed_count,
  cancelled_count: current7.cancelled_count,
  timeout_count: current7.timeout_count,
  authenticated_terminal_seed_count: current7.authenticated_terminal_seed_count,
  terminal_results_verified: current7.authenticated_terminal_seed_count,
  refused_terminal_count: current7.refused_terminal_count,
  failed_terminal_results_verified: current7.refused_terminal_count,
  feasible_pareto_count: current7.feasible_pareto_count,
};
const current7Root = {
  search_authority: {
    kind: "current7", configured: true, available: true,
    integrity_verified: true, legacy_role: "archive_only",
  },
  tier1_current7_search: current7,
  tier1_feedback_search: search,
  candidates: [
    {
      id: "current7-good", tier1_source_role: "current7",
      model_lane: "corrected_current7",
      model_id: "tier1-current7:current7-bundle",
      tier1_source_cohort_id: "current7-cohort",
      embedded_authenticated_result: true,
      production_eligible: false,
      fea_submission_approved: false,
      gui_launch_eligible: false,
      spec_status: "pass", volume_L: 800, total_loss_W: 5900,
    },
    {
      id: "legacy-stale", tier1_source_role: "primary",
      model_lane: "experimental_tier1_feedback",
      model_id: "tier1-feedback:legacy", spec_status: "pass",
      volume_L: 300, total_loss_W: 3000,
    },
  ],
  near_feasible_preview: [
    {
      id: "near-current7", source_role: "current7",
      source_cohort_id: "current7-cohort",
    },
    {
      id: "near-legacy", source_role: "primary",
      source_cohort_id: "legacy-cohort",
    },
  ],
  summary: { min_volume_L: 300, min_loss_W: 3000 },
};
const current7View = nsgaAuthorityView(current7Root);
const unavailableCurrent7Root = {
  ...current7Root,
  search_authority: {
    ...current7Root.search_authority,
    available: false,
    integrity_verified: false,
  },
  tier1_current7_search: { ...current7, available: false },
};
process.stdout.write(JSON.stringify({
  legacySelected: nsgaAuthoritySearch(
    { tier1_feedback_search: search },
  ) === search,
  legacyViewPreserved: nsgaAuthorityView(
    { tier1_feedback_search: search },
  ).tier1_feedback_search === search,
  matching: nsgaProgressMatchesDashboard(
    { tier1_feedback_search: search }, progress,
  ),
  wrongCohort: nsgaProgressMatchesDashboard(
    { tier1_feedback_search: search },
    { ...progress, sources: [source("cohort-old")] },
  ),
  wrongConstraint: nsgaProgressMatchesDashboard(
    { tier1_feedback_search: search },
    { ...progress, constraint_version: "res10k-t120" },
  ),
  unverified: nsgaProgressMatchesDashboard(
    { tier1_feedback_search: search },
    { ...progress, integrity_verified: false },
  ),
  current7Selected: nsgaAuthoritySearch(current7Root) === current7,
  current7CandidateIds: current7View.candidates.map((item) => item.id),
  current7NearIds: current7View.near_feasible_preview.map((item) => item.id),
  current7Summary: current7View.summary,
  unavailableCurrent7CandidateCount:
    nsgaAuthorityView(unavailableCurrent7Root).candidate_count,
  current7Matching: nsgaProgressMatchesDashboard(
    current7Root, current7Progress,
  ),
  staleLegacyRejected: nsgaProgressMatchesDashboard(
    current7Root, progress,
  ),
  wrongCurrent7Snapshot: nsgaProgressMatchesDashboard(
    current7Root,
    { ...current7Progress, authority_snapshot_sha256: "4".repeat(64) },
  ),
  wrongCurrent7Constraint: nsgaProgressMatchesDashboard(
    current7Root,
    { ...current7Progress, constraint_identity_sha256: "5".repeat(64) },
  ),
  wrongCurrent7Counts: nsgaProgressMatchesDashboard(
    current7Root,
    { ...current7Progress, running_count: current7.running_count - 1 },
  ),
  unavailableCurrent7DoesNotFallback: nsgaProgressMatchesDashboard(
    unavailableCurrent7Root,
    progress,
  ),
}));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(app_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )

    assert json.loads(completed.stdout) == {
        "legacySelected": True,
        "legacyViewPreserved": True,
        "matching": True,
        "wrongCohort": False,
        "wrongConstraint": False,
        "unverified": False,
        "current7Selected": True,
        "current7CandidateIds": ["current7-good"],
        "current7NearIds": ["near-current7"],
        "current7Summary": {
            "candidate_count": 1,
            "display_candidate_count": 1,
            "valid_candidate_count": 1,
            "min_volume_L": 800,
            "min_loss_W": 5900,
        },
        "unavailableCurrent7CandidateCount": 0,
        "current7Matching": True,
        "staleLegacyRejected": False,
        "wrongCurrent7Snapshot": False,
        "wrongCurrent7Constraint": False,
        "wrongCurrent7Counts": False,
        "unavailableCurrent7DoesNotFallback": False,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_nsga_near_violation_labels_keep_physical_units():
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = r"""
const path = require("node:path");
const { nearViolationLabel } = require(path.resolve(process.argv[1]));
process.stdout.write(JSON.stringify({
  resonance: nearViolationLabel({
    constraint: "half_magnetizing_resonance_minimum", amount: 1436.74,
  }),
  temperature: nearViolationLabel({
    constraint: "temperature_robust_limit:T_max_Tx", amount: 5.1197,
  }),
  passing: nearViolationLabel({
    constraint: "Llt_robust_band", amount: -0.1,
  }),
}));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(app_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    result = json.loads(completed.stdout)

    assert result["resonance"] == "공진 부족 +1.437 kHz"
    assert result["temperature"] == "Tx +5.12 °C"
    assert result["passing"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_nsga_near_candidate_dialog_uses_read_only_aliases_and_violations():
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = r"""
const path = require("node:path");
const { candidateDialogFacts } = require(path.resolve(process.argv[1]));
const result = candidateDialogFacts({
  id: "near-final",
  valid_pareto: false,
  gui_launch_eligible: false,
  pred_Llt_phys_uH: 21.416,
  robust_max_temperature_C: 147.3,
  violations: [
    { constraint: "Llt_robust_band", amount: 4.882 },
    { constraint: "temperature_robust_limit:T_max_core", amount: 47.3 },
  ],
});
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(app_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "isReadOnlyNear": True,
        "predictedLeakage": 21.416,
        "suppliedMaxTemperature": 147.3,
        "violationLabels": [
            "Llt robust +4.882 µH",
            "core +47.30 °C",
        ],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_nsga_scatter_keeps_near_candidates_in_diagnostic_series_only():
    app_js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = r"""
const path = require("node:path");
const { nsgaScatterSeries } = require(path.resolve(process.argv[1]));
const result = nsgaScatterSeries(
  [
    { id: "pareto-a", volume_L: 700, total_loss_W: 5800 },
    { id: "pareto-no-coordinate", volume_L: null, total_loss_W: 5700 },
  ],
  [
    {
      id: "near-safe", volume_L: "730.2", total_loss_W: "6100",
      valid_pareto: false, production_eligible: false,
      fea_submission_approved: false, gui_launch_eligible: false,
    },
    {
      id: "near-unsafe", volume_L: 680, total_loss_W: 5600,
      valid_pareto: false, production_eligible: true,
      fea_submission_approved: false, gui_launch_eligible: false,
    },
    {
      id: "not-near", volume_L: 650, total_loss_W: 5500,
      valid_pareto: true, production_eligible: false,
      fea_submission_approved: false, gui_launch_eligible: false,
    },
  ],
);
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(app_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    result = json.loads(completed.stdout)

    assert [(item["id"], item["chart_kind"]) for item in result["paretoPoints"]] == [
        ("pareto-a", "pareto"),
    ]
    assert [
        (item["id"], item["chart_kind"])
        for item in result["diagnosticPoints"]
    ] == [("near-safe", "near")]


def test_api_failure_is_section_local(campaign_root):
    class BrokenService:
        def dashboard(self):
            raise RuntimeError("broken artifact")

    client = TestClient(create_app(regression_root=campaign_root, service=BrokenService()))
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "broken artifact" in response.json()["error"]


def test_nsga_and_dashboard_join_latest_local_aedt_validation(tmp_path):
    class StubService:
        def nsga2(self):
            return {
                "available": True,
                "candidate_count": 2,
                "candidates": [{"id": "candidate-1"}, {"id": "candidate-2"}],
                "sealed_successor": {
                    "available": True,
                    "source_kind": "sealed_successor_evidence",
                    "canonical_candidate": False,
                    "terminal_result": False,
                    "candidate": {"candidate_sha256": "9" * 64},
                },
                "sealed_successors": {
                    "available": True,
                    "source_kind": "sealed_successor_evidence",
                    "canonical_candidate": False,
                    "terminal_result": False,
                    "pareto": False,
                    "production": False,
                    "fea": False,
                    "candidate_count": 2,
                    "candidates": [
                        {
                            "evidence_id": "exact-9375",
                            "terminal_result": False,
                            "pareto": False,
                            "production": False,
                            "fea": False,
                        },
                        {
                            "evidence_id": "interior-be52",
                            "terminal_result": False,
                            "pareto": False,
                            "production": False,
                            "fea": False,
                        },
                    ],
                },
            }

        def dashboard(self):
            return {"nsga2": self.nsga2(), "data": {"total_rows": 10}}

    class StubLauncher:
        def validation_index(self):
            return {
                "candidate-1": {
                    "launch_id": "launch-1",
                    "state": "completed_held",
                    "retained_aedt": True,
                    "fea_validation": {
                        "status": "completed",
                        "result": {"Llt_phys": 27.5},
                    },
                }
            }

    client = TestClient(create_app(
        regression_root=tmp_path,
        service=StubService(),
        local_gui_launcher=StubLauncher(),
    ))
    for endpoint in ("/api/nsga2", "/api/dashboard"):
        payload = client.get(endpoint).json()
        nsga = payload["nsga2"] if endpoint.endswith("dashboard") else payload
        first, second = nsga["candidates"]
        assert first["local_aedt_validation"]["launch_id"] == "launch-1"
        assert first["local_aedt_validation"]["retained_aedt"] is True
        assert second["local_aedt_validation"] is None
        assert nsga["candidate_count"] == 2
        assert len(nsga["candidates"]) == 2
        assert nsga["sealed_successor"] == {
            "available": True,
            "source_kind": "sealed_successor_evidence",
            "canonical_candidate": False,
            "terminal_result": False,
            "candidate": {"candidate_sha256": "9" * 64},
        }
        assert nsga["sealed_successors"]["candidate_count"] == 2
        assert [
            item["evidence_id"]
            for item in nsga["sealed_successors"]["candidates"]
        ] == ["exact-9375", "interior-be52"]
        assert all(
            item["terminal_result"] is False
            and item["pareto"] is False
            and item["production"] is False
            and item["fea"] is False
            for item in nsga["sealed_successors"]["candidates"]
        )


def test_nsga_progress_and_local_launches_are_read_only_projections(tmp_path):
    canonical_nsga = {
        "available": True,
        "status": "running",
        "source_kind": "continuous_nsga_with_tier1_feedback",
        "candidate_count": 3,
        "valid_candidate_count": 1,
        "tier1_feedback_search": {
            "available": True,
            "integrity_verified": True,
            "running_count": 7,
            "queued_count": 11,
            "attaching_count": 2,
            "active_plus_queued": 20,
            "completed_count": 4,
            "terminal_results_verified": 4,
            "failed_terminal_results_verified": 1,
            "feasible_pareto_count": 3,
            "candidate_preview_count": 3,
            "candidate_preview_limit": 128,
            "candidate_preview_truncated": False,
            "near_feasible_count": 9,
            "near_feasible_preview_count": 9,
            "near_feasible_preview_limit": 64,
            "near_feasible_preview_truncated": False,
            "coherent_snapshot_verified": True,
            "coherent_snapshot_attempts": 2,
            "source_count": 2,
            "supplemental_source_count": 1,
            "configured_source_count": 2,
            "rejected_source_count": 0,
            "all_configured_sources_verified": True,
            "source_aggregation": "authenticated_source_pareto_union",
            "sources": [
                {
                    "role": "primary",
                    "label": "main",
                    "accepted": True,
                    "search_count": 15,
                    "completed_count": 4,
                },
                {
                    "role": "supplemental",
                    "label": "deep-p320-g600-v1",
                    "accepted": True,
                    "search_count": 5,
                    "completed_count": 0,
                },
            ],
            "constraint_version": "constraint-v2",
            "constraints": {
                "T_limit_C": 120.0,
                "resonance_min_Hz": 10_000.0,
            },
            "temperature_constraint_contract": {
                "target_count": 11,
                "robust_upper_bound_C": 120.0,
            },
            "model_manifest_sha256": "1" * 64,
            "deployment_model_manifest_sha256": "2" * 64,
            "updated_at": "2026-07-18T15:00:00+09:00",
            "warnings": [],
        },
        "candidates": [],
    }

    class StubService:
        def nsga2(self):
            return canonical_nsga

    class StubLauncher:
        def snapshot(self, candidate_id=None):
            return {
                "schema_version": 1,
                "available": True,
                "backend": "standalone",
                "candidate_id": candidate_id,
                "launches": [],
            }

        def validation_index(self):
            return {}

    client = TestClient(create_app(
        regression_root=tmp_path,
        service=StubService(),
        local_gui_launcher=StubLauncher(),
    ))
    progress = client.get("/api/nsga2/progress").json()
    assert progress["source_endpoint"] == "/api/nsga2"
    assert progress["integrity_verified"] is True
    assert progress["counters_consistent"] is True
    assert progress["active_plus_queued"] == 20
    assert progress["terminal_results_verified"] == 4
    assert progress["failed_terminal_results_verified"] == 1
    assert progress["feasible_pareto_count"] == 3
    assert progress["candidate_preview_count"] == 3
    assert progress["candidate_preview_limit"] == 128
    assert progress["candidate_preview_truncated"] is False
    assert progress["candidate_preview_consistent"] is True
    assert progress["near_feasible_preview_count"] == 9
    assert progress["coherent_snapshot_attempts"] == 2
    assert progress["source_count"] == 2
    assert progress["supplemental_source_count"] == 1
    assert progress["all_configured_sources_verified"] is True
    assert progress["sources"][1]["label"] == "deep-p320-g600-v1"
    assert progress["constraints"]["resonance_min_Hz"] == 10_000.0
    assert progress["temperature_constraint_contract"]["target_count"] == 11
    canonical_nsga["tier1_feedback_search"][
        "candidate_preview_truncated"
    ] = True
    inconsistent_preview = client.get("/api/nsga2/progress").json()
    assert inconsistent_preview["available"] is False
    assert inconsistent_preview["integrity_verified"] is False
    assert inconsistent_preview["candidate_preview_consistent"] is False
    canonical_nsga["tier1_feedback_search"][
        "candidate_preview_truncated"
    ] = False

    launches = client.get(
        "/api/local-aedt-gui/launches?candidate_id=candidate-1"
    ).json()
    status = client.get(
        "/api/local-aedt-gui/status?candidate_id=candidate-1"
    ).json()
    assert launches == status
    assert launches["backend"] == "standalone"
    assert launches["candidate_id"] == "candidate-1"


def test_nsga_progress_fails_closed_on_counter_mismatch(tmp_path):
    class StubService:
        def nsga2(self):
            return {
                "available": True,
                "tier1_feedback_search": {
                    "available": True,
                    "integrity_verified": True,
                    "running_count": 1,
                    "queued_count": 2,
                    "attaching_count": 3,
                    "active_plus_queued": 99,
                    "completed_count": 4,
                    "terminal_results_verified": 4,
                    "feasible_pareto_count": 0,
                    "near_feasible_count": 1,
                },
            }

    payload = TestClient(create_app(
        regression_root=tmp_path,
        service=StubService(),
    )).get("/api/nsga2/progress").json()
    assert payload["available"] is False
    assert payload["integrity_verified"] is False
    assert payload["counters_consistent"] is False


def test_dashboard_and_status_expose_refill_controller(artifact_service):
    expected = {
        "available": True,
        "last_tick_at": "2026-07-13T12:30:00+09:00",
        "action": "no_refill_needed",
        "active_project_tasks_before": 300,
        "accepted_or_reconciled_count": 0,
        "generation_id": "restart-v3-generation",
        "concurrency_target": 300,
    }

    class StubRefillController:
        def snapshot(self):
            return dict(expected)

    artifact_service.refill_controller = StubRefillController()
    client = TestClient(create_app(service=artifact_service))

    assert client.get("/api/dashboard").json()["refill_controller"] == expected
    assert client.get("/api/status").json()["refill_controller"] == expected


def test_local_operator_can_set_versioned_drain_simulation_policy(
        artifact_service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "regression_260707.monitoring.app.CAMPAIGN_MUTATION_LOCK_PATH",
        tmp_path / "campaign-mutation.lock",
    )
    client = _addressed_client(
        create_app(service=artifact_service),
        base_url="http://127.0.0.1:8010",
        client=("127.0.0.1", 51000),
    )
    response = client.patch(
        "/api/operator/simulation-policy",
        headers={
            "Content-Type": "application/json",
            "X-MFT-Operator-Control": "simulation-policy-v1",
            "Origin": "http://127.0.0.1:8010",
        },
        json={
            "desired_simulations": 500,
            "expected_revision": 7,
            "scale_down_mode": "drain",
        },
    )

    assert response.status_code == 200
    assert response.json()["updated"] is True
    assert response.json()["project"] == "MFT_1MW_2026v1"
    assert response.json()["desired_simulations"] == 500
    assert response.json()["policy_revision"] == 8
    assert artifact_service.scheduler.parallel_target == 500


def test_simulation_policy_control_rejects_csrf_remote_and_invalid_requests(
        artifact_service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "regression_260707.monitoring.app.CAMPAIGN_MUTATION_LOCK_PATH",
        tmp_path / "campaign-mutation.lock",
    )
    app = create_app(service=artifact_service)
    local = _addressed_client(
        app,
        base_url="http://127.0.0.1:8010",
        client=("127.0.0.1", 51001),
    )
    valid_headers = {
        "Content-Type": "application/json",
        "X-MFT-Operator-Control": "simulation-policy-v1",
    }
    valid_payload = {
        "desired_simulations": 300,
        "expected_revision": 7,
        "scale_down_mode": "drain",
    }

    assert local.patch(
        "/api/operator/simulation-policy",
        headers={"Content-Type": "application/json"},
        json=valid_payload,
    ).status_code == 403
    assert local.patch(
        "/api/operator/simulation-policy",
        headers={**valid_headers, "Origin": "https://attacker.invalid"},
        json=valid_payload,
    ).status_code == 403
    assert local.patch(
        "/api/operator/simulation-policy",
        headers={**valid_headers, "Host": "monitor.public.invalid"},
        json=valid_payload,
    ).status_code == 403
    for invalid in (501, -1, 1.5, True, "300"):
        response = local.patch(
            "/api/operator/simulation-policy",
            headers=valid_headers,
            json={**valid_payload, "desired_simulations": invalid},
        )
        assert response.status_code == 422

    remote = _addressed_client(
        app,
        base_url="http://127.0.0.1:8010",
        client=("192.0.2.10", 51002),
    )
    assert remote.patch(
        "/api/operator/simulation-policy",
        headers=valid_headers,
        json=valid_payload,
    ).status_code == 403


def test_simulation_policy_allows_allowlisted_trusted_lan(
        artifact_service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "regression_260707.monitoring.app.CAMPAIGN_MUTATION_LOCK_PATH",
        tmp_path / "campaign-mutation.lock",
    )
    monkeypatch.setenv("MFT_MONITOR_OPERATOR_HOSTS", "monitor.local,192.168.0.37")
    client = _addressed_client(
        create_app(service=artifact_service),
        base_url="http://192.168.0.37:8010",
        client=("192.168.0.18", 51003),
    )
    response = client.patch(
        "/api/operator/simulation-policy",
        headers={
            "Content-Type": "application/json",
            "X-MFT-Operator-Control": "simulation-policy-v1",
            "Origin": "http://192.168.0.37:8010",
        },
        json={
            "desired_simulations": 0,
            "expected_revision": 7,
            "scale_down_mode": "drain",
        },
    )

    assert response.status_code == 200
    assert response.json()["desired_simulations"] == 0


def test_simulation_policy_rejects_stale_browser_revision(
        artifact_service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "regression_260707.monitoring.app.CAMPAIGN_MUTATION_LOCK_PATH",
        tmp_path / "campaign-mutation.lock",
    )
    client = _addressed_client(
        create_app(service=artifact_service),
        base_url="http://127.0.0.1:8010",
        client=("127.0.0.1", 51004),
    )
    response = client.patch(
        "/api/operator/simulation-policy",
        headers={
            "Content-Type": "application/json",
            "X-MFT-Operator-Control": "simulation-policy-v1",
        },
        json={
            "desired_simulations": 500,
            "expected_revision": 6,
            "scale_down_mode": "drain",
        },
    )

    assert response.status_code == 409
    assert artifact_service.scheduler.parallel_target == 300
    assert artifact_service.scheduler.policy_revision == 7
