(() => {
  "use strict";

  const hasDOM = typeof document !== "undefined";
  const $ = (selector) => document.querySelector(selector);
  const svgNS = "http://www.w3.org/2000/svg";
  const refreshSeconds = hasDOM ? Number(document.body.dataset.refreshSeconds || 20) : 20;
  // Bound each transport without coupling the dashboard's first render to sidecars.
  const dashboardRequestTimeoutMs = 120000;
  const nsgaProgressRequestTimeoutMs = 120000;
  const localGuiRequestTimeoutMs = 30000;
  const codexWorkRequestTimeoutMs = 5000;
  const state = {
    dashboard: null, selectedModel: null, selectedModelData: null, refreshing: false,
    historyMetric: "r2",
    parityCache: new Map(), parityRequest: 0,
    parallelTargetDirty: false, updatingParallelTarget: false,
    selectedCandidate: null, localGuiBusy: false, localGuiPollTimer: null,
    nsgaProgress: null, localGuiLaunches: null, sealedSuccessor: null,
    sealedSuccessors: [], sealedSuccessorId: null,
    nsgaPayload: null, selectedNsgaGenerationId: null,
    nsgaGenerationCache: new Map(),
    nsgaGenerationRequests: new Set(),
    nsgaProgressRequest: null, localGuiLaunchesRequest: null,
    codexWork: null, codexWorkRequest: null,
  };

  const labels = {
    loading: "불러오는 중", active: "진행 중", warning: "주의", error: "오류", idle: "대기",
    complete: "완료", waiting: "대기", trained: "학습 완료", stale: "재학습 필요",
    attention: "성능 주의", checkpoint: "체크포인트 CV", not_trained: "학습 전",
    invalid_provenance: "교정 재학습 대기",
    parity_overlay: "교정 OOF overlay",
    pass: "PASS", fail: "FAIL", unknown: "확인 불가",
  };
  labels.provisional_pass_complete_full_pending = "PROVISIONAL PASS";
  const checkLabels = {
    llt: "누설 인덕턴스", temperature: "최고 온도",
    bfield: "설계 B = V/(4fNAe)",
    insulation: "최소 절연 간격",
    core_group: "코어 조 수",
    primary_thickness: "1차 도체 두께",
    resonance: "0.5×Lm 최저 공진",
    size_width: "외형 폭 W",
    size_length: "외형 길이 L",
    size_height: "외형 높이 H",
    convergence: "수렴오차 ≤1.5%", full_model: "Full model",
  };
  const historyMetrics = {
    r2: { label: "CV R²", field: "r2", color: "#55aaff", digits: 3 },
    mape_pct: { label: "CV MAPE", field: "mape_pct", color: "#ffbb55", digits: 2, suffix: "%" },
    rmse: { label: "CV RMSE", field: "rmse", color: "#26d7c7", digits: 3 },
    p90_ape_pct: { label: "CV P90 APE", field: "p90_ape_pct", color: "#ff626d", digits: 2, suffix: "%" },
  };

  function setText(selector, value) {
    const node = $(selector);
    if (node) node.textContent = value == null || value === "" ? "—" : String(value);
  }

  function number(value, digits = 0) {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    return Number(value).toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function compact(value, unit = "") {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    const n = Number(value);
    const digits = Math.abs(n) < 10 ? 3 : Math.abs(n) < 100 ? 2 : 1;
    return `${number(n, digits)}${unit ? ` ${unit}` : ""}`;
  }

  function candidateSources(candidate) {
    if (!candidate || typeof candidate !== "object") return [];
    return [
      candidate,
      candidate.report,
      candidate.full_row,
      candidate.row,
      candidate.details,
      candidate.parameters,
    ].filter((source) => source && typeof source === "object" && !Array.isArray(source));
  }

  function candidateValue(candidate, ...keys) {
    const sources = candidateSources(candidate);
    for (const key of keys) {
      for (const source of sources) {
        if (!Object.prototype.hasOwnProperty.call(source, key)) continue;
        const value = source[key];
        if (value != null && value !== "") return value;
      }
    }
    return null;
  }

  function candidateNumber(candidate, ...keys) {
    const sources = candidateSources(candidate);
    for (const key of keys) {
      for (const source of sources) {
        if (!Object.prototype.hasOwnProperty.call(source, key)) continue;
        const value = source[key];
        const numericValue = typeof value === "string" ? value.trim() : value;
        if (numericValue !== "" && typeof numericValue !== "boolean"
            && numericValue != null && Number.isFinite(Number(numericValue))) return Number(numericValue);
      }
    }
    return null;
  }

  function candidateTemperature(candidate, target, ...aliases) {
    const containers = candidateSources(candidate).flatMap((source) => [
      source.pred_temperatures_C,
      source.temperatures_C,
    ]).filter((source) => source && typeof source === "object" && !Array.isArray(source));
    for (const key of [target, ...aliases]) {
      for (const source of containers) {
        if (!Object.prototype.hasOwnProperty.call(source, key)) continue;
        const value = source[key];
        const numericValue = typeof value === "string" ? value.trim() : value;
        if (numericValue !== "" && typeof numericValue !== "boolean"
            && numericValue != null && Number.isFinite(Number(numericValue))) return Number(numericValue);
      }
    }
    return candidateNumber(candidate, `pred_${target}`, target, ...aliases);
  }

  function scaledCandidateNumber(value, multiplier, unit) {
    return value == null || !Number.isFinite(Number(value))
      ? "—"
      : compact(Number(value) * multiplier, unit);
  }

  function candidateCapacitance(candidate, windingPair) {
    const nanofarads = candidateNumber(
      candidate,
      `pred_C_${windingPair}_nF`,
      `C_${windingPair}_nF`,
    );
    if (nanofarads != null) return compact(nanofarads, "nF");
    return scaledCandidateNumber(candidateNumber(
      candidate,
      `pred_C_${windingPair}_F`,
      `C_${windingPair}_F`,
    ), 1e9, "nF");
  }

  function candidateResonance(candidate, mode) {
    const kilohertz = candidateNumber(
      candidate,
      `pred_f_res_${mode}_kHz`,
      `f_res_${mode}_kHz`,
    );
    if (kilohertz != null) return compact(kilohertz, "kHz");
    return scaledCandidateNumber(candidateNumber(
      candidate,
      `pred_f_res_${mode}_Hz`,
      `f_res_${mode}_Hz`,
    ), 1e-3, "kHz");
  }

  function duration(value) {
    if (value == null || value === "" || !Number.isFinite(Number(value)) || Number(value) < 0) return "—";
    const seconds = Math.round(Number(value));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    if (hours) return `${hours}h ${minutes}m ${remainder}s`;
    if (minutes) return `${minutes}m ${remainder}s`;
    return `${remainder}s`;
  }

  function dateTime(value, withDate = true) {
    if (!value) return "—";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return new Intl.DateTimeFormat("ko-KR", {
      ...(withDate ? { month: "2-digit", day: "2-digit" } : {}),
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).format(parsed);
  }

  function elapsed(minutes) {
    if (minutes == null || !Number.isFinite(Number(minutes))) return "갱신 시각 없음";
    if (minutes < 1) return "방금 갱신";
    if (minutes < 60) return `${Math.floor(minutes)}분 전 갱신`;
    return `${number(minutes / 60, 1)}시간 전 갱신`;
  }

  function relativeTickTime(value) {
    if (!value) return "시각 확인 불가";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "시각 확인 불가";
    const minutes = Math.max(0, Math.floor((Date.now() - parsed.getTime()) / 60000));
    return minutes < 1 ? "방금" : `${number(minutes)}분 전`;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function hasNumber(value) {
    return value != null && value !== "" && Number.isFinite(Number(value));
  }

  function count(value, suffix = "개") {
    return hasNumber(value) ? `${number(value)}${suffix}` : "—";
  }

  function ratio(numerator, denominator, suffix = "") {
    if (!hasNumber(numerator) || !hasNumber(denominator)) return "—";
    return `${number(numerator)} / ${number(denominator)}${suffix}`;
  }

  function rangeSummary(stats = {}, unit = "", digits = 3) {
    const unitSuffix = unit ? ` ${unit}` : "";
    const median = hasNumber(stats.median) ? `${number(stats.median, digits)}${unitSuffix}` : "—";
    const minimum = hasNumber(stats.min) ? `${number(stats.min, digits)}${unitSuffix}` : "—";
    const maximum = hasNumber(stats.max) ? `${number(stats.max, digits)}${unitSuffix}` : "—";
    return { median, detail: `최소 ${minimum} · 최대 ${maximum} · n=${number(stats.sample_count)}` };
  }

  function timingCell(timing = {}) {
    const cell = element("td", "simulation-timing");
    const grid = element("div", "timing-grid");
    [
      ["matrix", "Matrix"], ["loss", "Loss"],
      ["icepak", "Icepak"], ["total", "Total"],
    ].forEach(([key, label]) => {
      const item = element("span", `timing-item${key === "total" ? " total" : ""}`);
      item.append(element("b", "", label), element("span", "", duration(timing[key])));
      grid.append(item);
    });
    cell.append(grid);
    return cell;
  }

  function svgElement(tag, attrs = {}) {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function renderOverall(payload) {
    const status = payload.status || {};
    const overall = status.overall || "warning";
    const pill = $("#overall-status");
    pill.className = `status-pill status-${overall}`;
    pill.replaceChildren(element("i"), document.createTextNode(` ${labels[overall] || overall}`));
    setText("#current-stage", status.current_stage_label || "상태 확인 불가");
    setText("#last-refresh", `${dateTime(payload.generated_at)} 갱신`);

    const alerts = Array.isArray(status.warnings) ? status.warnings : [];
    const panel = $("#alert-panel");
    const list = $("#alert-list");
    list.replaceChildren();
    alerts.slice(0, 8).forEach((message) => list.append(element("li", "", message)));
    panel.classList.toggle("hidden", alerts.length === 0);
    renderPipeline(status.stages || []);
  }

  function renderPipeline(stages) {
    const pipeline = $("#pipeline");
    pipeline.replaceChildren();
    stages.forEach((stage, index) => {
      const item = element("li", stage.state || "waiting");
      item.append(element("em", "", String(index + 1).padStart(2, "0")));
      item.append(element("b", "", stage.label));
      item.append(element("span", "", stage.detail || labels[stage.state] || "—"));
      pipeline.append(item);
    });
  }

  function compactGeneration(value) {
    if (!value) return "—";
    const normalized = String(value).replaceAll("\\", "/").replace(/\/$/, "");
    const parts = normalized.split("/").filter(Boolean);
    if (parts.length <= 2) return normalized;
    return `…/${parts.slice(-2).join("/")}`;
  }

  function pipelineStateLabel(value) {
    return ({
      healthy: "정상", degraded: "주의", offline: "중지",
      alive: "실행 중", stale: "stale", missing: "없음", invalid: "오류",
      unknown: "확인 불가", running: "실행 중", waiting: "대기",
      retrying: "재시도", succeeded: "완료", failed: "실패", idle: "idle", unavailable: "읽기 실패",
    })[value] || value || "확인 불가";
  }

  function pipelineStateClass(value) {
    if (["healthy", "alive", "running", "succeeded"].includes(value)) return "pass";
    if (["offline", "stale", "invalid", "failed"].includes(value)) return "fail";
    return "unknown";
  }

  function renderPipelineRole(name, role = {}) {
    const chip = $(`#pipeline-${name}-state`);
    const roleState = role.status || "missing";
    chip.className = `state-chip ${pipelineStateClass(roleState)}`;
    chip.textContent = pipelineStateLabel(roleState);
    const process = role.pid
      ? `PID ${role.pid} · ${dateTime(role.started_at)} · ${duration(role.elapsed_seconds)}`
      : "PID 없음";
    setText(`#pipeline-${name}-process`, process);
    const activity = role.last_activity_at
      ? `${dateTime(role.last_activity_at)} · ${duration(role.activity_age_seconds)} 전`
      : "활동 기록 없음";
    setText(`#pipeline-${name}-heartbeat`, activity);
    const errorNode = $(`#pipeline-${name}-error`);
    errorNode.textContent = role.last_error || role.error || "없음";
    errorNode.classList.toggle("pipeline-error", Boolean(role.last_error || role.error));
  }

  function renderPipelineRevision(selector, value, exact = true) {
    const node = $(selector);
    node.textContent = value ? String(value).slice(0, 12) : "—";
    node.title = value || "";
    node.classList.toggle("revision-invalid", Boolean(value) && !exact);
  }

  function renderContinuousPipeline(pipeline = {}) {
    const health = pipeline.health || "offline";
    const status = $("#continuous-pipeline-state");
    status.className = `status-pill status-${health === "healthy" ? "active" : health === "offline" ? "error" : "warning"}`;
    status.replaceChildren(element("i"), document.createTextNode(` ${pipelineStateLabel(health)}`));
    setText("#continuous-pipeline-root", pipeline.root || "—");
    renderPipelineRole("controller", pipeline.roles?.controller || {});
    renderPipelineRole("supervisor", pipeline.roles?.supervisor || {});

    const revisions = pipeline.revisions || {};
    renderPipelineRevision(
      "#pipeline-solver-revision",
      revisions.solver_revision,
      revisions.solver_revision_exact,
    );
    renderPipelineRevision(
      "#pipeline-library-revision",
      revisions.library_revision,
      revisions.library_revision_exact,
    );
    renderPipelineRevision(
      "#pipeline-verification-revision",
      revisions.verification_config_sha256,
      true,
    );
    const cohort = pipeline.cohort || {};
    const rawRows = cohort.raw_rows ?? cohort.current_raw_rows;
    const strictEmRows = cohort.strict_em_rows ?? cohort.current_strict_em_rows;
    const strictFullRows = cohort.strict_full_rows ?? cohort.current_strict_full_rows ?? 0;
    const emOnlyRows = cohort.em_only_rows ?? cohort.current_em_only_rows;
    const completeClassification = [rawRows, strictEmRows, strictFullRows, emOnlyRows]
      .every((value) => value != null && Number.isFinite(Number(value)));
    setText(
      "#pipeline-em-rows",
      completeClassification
        ? `${number(rawRows)} / ${number(strictEmRows)}`
        : "Parquet 분류 확인 중",
    );
    setText(
      "#pipeline-em-only-rows",
      completeClassification
        ? `${number(emOnlyRows)} (thermal incomplete)`
        : "—",
    );
    setText(
      "#pipeline-exact-rows",
      `${number(strictFullRows)} · train ${number(cohort.first_training_rows || 500)} / model ${number(cohort.model_activation_rows || 3000)} / tune ${number(cohort.first_tuning_rows || 4000)}`,
    );
    const cohortExplanation = $("#pipeline-cohort-explanation");
    if (completeClassification) {
      cohortExplanation.textContent = Number(emOnlyRows) > 0
        ? `${number(emOnlyRows)}개는 무효 데이터가 아닙니다. EM 기준을 통과했지만 열 결과가 불완전해 Full(EM+열) gate에서만 제외됩니다.`
        : "모든 EM 사용 가능 행이 열 품질 기준까지 통과했습니다.";
    } else {
      cohortExplanation.textContent = "Full 행은 manifest에서 확인됐으며, EM/열 분리 집계를 읽는 중입니다.";
    }
    cohortExplanation.classList.toggle("pipeline-cohort-audit-warning", Boolean(cohort.counts_warning));
    cohortExplanation.title = cohort.counts_warning || cohort.counts_error || cohort.counts_source || "";
    renderPipelineRevision(
      "#pipeline-dataset-generation",
      cohort.generation,
      cohort.available === true,
    );
    const external = pipeline.external_tuners || {};
    const externalProcesses = Array.isArray(external.processes) ? external.processes : [];
    const externalVerified = Number(external.validated_running_count || 0);
    const externalState = $("#pipeline-external-tuner-state");
    externalState.className = `state-chip ${externalVerified ? "pass" : externalProcesses.length ? "unknown" : external.available ? "pass" : "fail"}`;
    externalState.textContent = externalVerified
      ? `${externalVerified}개 활동 검증`
      : externalProcesses.length
        ? `${externalProcesses.length}개 확인 대기`
        : external.available ? "없음" : "확인 불가";
    setText(
      "#pipeline-external-tuner-detail",
      externalVerified
        ? "최근 CPU/I/O 증가가 확인되어 observed 병렬 lane에 포함됩니다. durable queue 수에는 포함되지 않습니다."
        : externalProcesses.length
          ? "PID는 있으나 최근 CPU/I/O 증가가 확인되기 전에는 병렬 lane으로 세지 않습니다."
        : external.error || "durable queue 밖의 Optuna 작업이 없습니다.",
    );
    const externalList = $("#pipeline-external-tuners");
    externalList.replaceChildren();
    externalProcesses.forEach((process) => {
      const dataset = compactGeneration(process.dataset);
      externalList.append(element(
        "li", "",
        `PID ${process.pid} · ${process.validated_running ? "활동 검증" : "검증 대기"} · CPU +${number(process.cpu_seconds_delta, 2)}s · read +${number(process.read_bytes_delta)}B · ${duration(process.elapsed_seconds)} · trials ${process.trials || "—"} · ${dataset}`,
      ));
    });

    const hpo = pipeline.experimental_shadow_training || {};
    const hpoState = $("#pipeline-hpo-state");
    hpoState.className = `state-chip ${hpo.validated_running ? "pass" : hpo.available ? "unknown" : "fail"}`;
    hpoState.textContent = hpo.validated_running
      ? "running · heartbeat verified"
      : hpo.available ? (hpo.state || "waiting") : "unavailable";
    setText(
      "#pipeline-hpo-progress",
      `targets ${number(hpo.completed_hpo_target_count || 0)} / ${number(hpo.selected_hpo_target_count || 0)}`
        + ` · processes ${number(hpo.active_hpo_processes || 0)}`
        + ` · batch ${number(hpo.active_hpo_batch || 0)} / ${number(hpo.hpo_batch_count || 0)}`,
    );
    setText(
      "#pipeline-hpo-detail",
      `${hpo.wave_phase || "phase unavailable"} · rows ${number(hpo.observed_strict_full_rows || 0)}`
        + ` · heartbeat age ${hpo.age_seconds == null ? "—" : duration(hpo.age_seconds)}`
        + `${hpo.error ? ` · ${hpo.error}` : ""}`,
    );

    const parallel = pipeline.parallel || {};
    const queue = pipeline.queue || {};
    const counts = queue.counts || {};
    setText(
      "#pipeline-running-lanes",
      `${number(parallel.running_lane_count || 0)} observed (${number(parallel.durable_running_lane_count || 0)} durable + ${number(parallel.external_running_lane_count || 0)} external)`,
    );
    setText(
      "#pipeline-active-lanes",
      `${number(parallel.active_lane_count || 0)} observed (${number(parallel.durable_active_lane_count || 0)} durable)`,
    );
    setText("#pipeline-queued-jobs", number((counts.queued || 0) + (counts.retry_wait || 0)));
    setText("#pipeline-running-jobs", number(counts.running || 0));
    setText("#pipeline-succeeded-jobs", number(counts.succeeded || 0));
    setText("#pipeline-failed-jobs", number(counts.failed || 0));
    const runningNames = Array.isArray(parallel.running_lanes) ? parallel.running_lanes : [];
    const roleSummary = pipeline.roles?.controller?.alive && pipeline.roles?.supervisor?.alive
      ? "controller와 supervisor가 모두 실행 중"
      : "controller/supervisor 실행 상태 확인 필요";
    const parallelSummary = parallel.parallel_work_confirmed
      ? `${runningNames.length}개 lane 실제 동시 실행 확인 (${runningNames.join(", ")})`
      : runningNames.length
        ? `현재 ${runningNames[0]} lane 실행 중; 다른 lane은 조건/의존성 대기`
        : "현재 running job 없음; queue 조건 또는 데이터 checkpoint 대기";
    setText("#continuous-pipeline-detail", `${roleSummary} · ${parallelSummary}`);

    const body = $("#continuous-pipeline-lanes");
    body.replaceChildren();
    const lanes = Array.isArray(pipeline.lanes) ? pipeline.lanes : [];
    lanes.forEach((lane) => {
      const row = element("tr", `pipeline-lane-${lane.health || "unavailable"}`);
      const identity = element("td", "pipeline-lane-identity");
      identity.append(element("b", "", lane.label || lane.job_type));
      identity.append(element("code", "", lane.job_type || "—"));
      row.append(identity);
      const healthCell = element("td");
      healthCell.append(element(
        "span",
        `state-chip ${pipelineStateClass(lane.health)}`,
        pipelineStateLabel(lane.health),
      ));
      row.append(healthCell);
      const laneCounts = lane.counts || {};
      row.append(element(
        "td",
        "pipeline-counts mono",
        `${number((laneCounts.queued || 0) + (laneCounts.retry_wait || 0))} / ${number(laneCounts.running || 0)} / ${number(laneCounts.succeeded || 0)} / ${number(laneCounts.failed || 0)}`,
      ));
      const job = lane.current_job;
      row.append(element(
        "td", "mono",
        job ? `#${job.id} · ${job.attempt}/${job.max_attempts}` : "—",
      ));
      row.append(element("td", "pipeline-time", job ? dateTime(job.started_at) : "—"));
      const heartbeat = job?.heartbeat_at
        ? `${dateTime(job.heartbeat_at)} · ${duration(job.heartbeat_age_seconds)} 전`
        : "—";
      const heartbeatCell = element(
        "td",
        job?.heartbeat_stale ? "pipeline-heartbeat stale" : "pipeline-heartbeat",
        heartbeat,
      );
      row.append(heartbeatCell);
      row.append(element("td", "mono", job ? duration(job.elapsed_seconds) : "—"));
      const evidence = element("td", "pipeline-evidence");
      const generations = element("div", "pipeline-generations");
      const input = element("code", "", compactGeneration(job?.input_generation));
      input.title = job?.input_generation || "";
      const output = element("code", "", compactGeneration(job?.output_generation));
      output.title = job?.output_generation || "";
      generations.append(input, element("span", "", "→"), output);
      const prerequisite = lane.prerequisite || {};
      if (prerequisite.reason) {
        evidence.append(element(
          "small",
          prerequisite.ready ? "pipeline-prerequisite ready" : "pipeline-prerequisite blocked",
          prerequisite.reason,
        ));
      }
      evidence.append(generations);
      const lastError = job?.terminal_reason || lane.last_error?.reason;
      if (lastError) {
        const error = element("small", "pipeline-error", lastError);
        error.title = lastError;
        evidence.append(error);
      } else {
        evidence.append(element("small", "pipeline-no-error", "오류 없음"));
      }
      row.append(evidence);
      body.append(row);
    });
    $("#continuous-pipeline-empty").classList.toggle(
      "hidden", queue.available === true && lanes.length > 0,
    );
    $(".continuous-lane-scroll").classList.toggle("hidden", lanes.length === 0);
  }

  function renderData(data) {
    const countBases = data.count_bases && typeof data.count_bases === "object"
      ? data.count_bases : {};
    const training = data.training_cohort && typeof data.training_cohort === "object"
      ? data.training_cohort : {};
    const eligible = data.latest_eligible_cohort && typeof data.latest_eligible_cohort === "object"
      ? data.latest_eligible_cohort : {};
    const trainingRows = eligible.available === true ? eligible.strict_full_rows : null;
    const completedTraining = data.latest_completed_training?.snapshot || {};
    const activeTraining = data.active_training_job || {};
    const activeModel = data.active_model || {};
    const compactRevision = (value) => {
      const text = String(value || "");
      return text ? text.slice(0, 10) : "—";
    };
    setText("#data-raw-total", `${number(data.raw_total_rows)}개`);
    setText("#data-raw-basis", `basis: ${countBases.raw_total || "—"}`);
    setText("#data-training-total", trainingRows == null ? "—" : `${number(trainingRows)}개`);
    setText(
      "#data-training-basis",
      `basis: ${training.count_basis || countBases.training_strict_full || "—"}`,
    );
    setText(
      "#data-training-revisions",
      `solver/library: ${compactRevision(training.solver_revision)} / ${compactRevision(training.library_revision)}`,
    );
    setText("#data-active-em", `${number(data.em_valid_rows)}개`);
    setText(
      "#data-active-em-basis",
      `basis: ${countBases.active_physics_strict_em || "—"} · raw ${number(data.revision_raw_rows)}개`,
    );
    setText("#data-total", `${number(data.total_rows)}개`);
    const snapshotMatchesActive = Boolean(
      activeTraining.input_generation
      && completedTraining.generation
      && activeTraining.input_generation === completedTraining.generation
    );
    setText(
      "#data-active-training",
      activeTraining.id ? `#${activeTraining.id} · ${activeTraining.state || "unknown"}` : "없음",
    );
    setText(
      "#data-active-training-detail",
      activeTraining.id
        ? `snapshot: ${snapshotMatchesActive ? number(completedTraining.strict_full_rows) : "검증 대기"} full · ${compactGeneration(activeTraining.input_generation)}`
        : "snapshot: —",
    );
    setText(
      "#data-completed-training",
      completedTraining.available === true
        ? `${number(completedTraining.strict_full_rows)} full` : "—",
    );
    setText(
      "#data-completed-training-detail",
      `snapshot: ${compactGeneration(completedTraining.generation)} · ${dateTime(completedTraining.updated_at)}`,
    );
    setText(
      "#data-active-model",
      activeModel.verified === true
        ? (activeModel.production_active ? "active" : activeModel.state || "awaiting")
        : "unverified",
    );
    setText(
      "#data-active-model-detail",
      `cohort: ${compactGeneration(activeModel.dataset_generation || activeModel.model_generation || activeModel.dataset_sha256)} · ${activeModel.strict_full_rows == null ? "—" : `${number(activeModel.strict_full_rows)} full`}`,
    );
    const physicsRevision = String(data.current_physics_data_revision || "");
    const physicsLabel = physicsRevision.length > 24 ? `${physicsRevision.slice(0, 24)}…` : (physicsRevision || "—");
    setText(
      "#data-strict-detail",
      `basis: ${data.count_basis || countBases.active_physics_strict_full || "—"} · physics ${physicsLabel}`,
    );
    const memberShas = Array.isArray(data.member_git_hash_shorts) ? data.member_git_hash_shorts : [];
    setText("#data-member-shas", `SHA: ${memberShas.length ? memberShas.join(", ") : "—"}`);
    setText("#data-throughput", `+${number(data.throughput_1h)}개`);
    setText("#data-throughput-detail", `24시간 +${number(data.added_24h)} · 유효속도 ${number(data.effective_hourly_rate, 1)}/h`);
    setText("#data-eta", data.eta_3000 ? dateTime(data.eta_3000) : "산정 불가");
    setText("#data-stall", data.eta_hours != null ? `약 ${number(data.eta_hours, 1)}시간 후` : "최근 처리량이 없습니다");
    setText("#data-freshness", elapsed(data.stalled_minutes));
    $("#data-freshness").classList.toggle("stale", Boolean(data.stalled));
    setText("#latest-revision", physicsRevision ? `physics ${physicsLabel}` : "physics —");
    setText("#goal-label", `${number(data.total_rows)} / ${number(data.goal)}`);
    setText("#stretch-label", `${number(data.total_rows)} / ${number(data.stretch_goal)}+`);
    $("#goal-progress").style.width = `${Math.max(0, Math.min(100, data.goal_progress_pct || 0))}%`;
    $("#stretch-progress").style.width = `${Math.max(0, Math.min(100, data.stretch_progress_pct || 0))}%`;
    setText("#data-24h", `+${number(data.added_24h)}개`);
    setText("#data-remaining", `${number(data.remaining_to_goal)}개`);
    setText("#revision-mismatch", data.rows_not_current_physics_revision == null ? "—" : `${number(data.rows_not_current_physics_revision)}개`);
    setText("#collector-nodata", data.collector?.no_data_tasks == null ? "—" : `${number(data.collector.no_data_tasks)}건`);
    const timing = data.simulation_timing || {};
    const timingStages = timing.stages || {};
    const timingActive = timing.active_cohort || data.active_cohort || {};
    const timingCohortLabel = timing.cohort_label || timingActive.label || "활성 코호트 확인 중";
    const timingWindowRows = Number.isFinite(Number(timing.window_rows)) ? timing.window_rows : 0;
    const timingWindowLimit = Number.isFinite(Number(timing.window_limit_rows)) ? timing.window_limit_rows : 100;
    setText(
      "#stage-timing-basis",
      timingActive.available === false
        ? timingCohortLabel
        : `${timingCohortLabel} 기준 · solver 결과의 실제 timing 필드`,
    );
    setText(
      "#stage-timing-window",
      timingActive.available === false
        ? timingCohortLabel
        : `${timingCohortLabel} · n=${number(timingWindowRows)} (최근 최대 ${number(timingWindowLimit)}행)`,
    );
    const timingEmpty = $("#stage-timing-empty");
    timingEmpty.textContent = timingActive.available === false
      ? timingCohortLabel
      : "활성 코호트 타이밍 데이터 없음";
    timingEmpty.classList.toggle("hidden", Boolean(timing.available));
    ["matrix", "loss", "electrostatic", "icepak", "total"].forEach((key) => {
      const stage = timingStages[key] || {};
      setText(`#stage-time-${key}-mean`, duration(stage.mean_seconds));
      setText(
        `#stage-time-${key}-detail`,
        `중앙값 ${duration(stage.median_seconds)} · n=${number(stage.sample_count)}`,
      );
    });
    lineChart($("#data-chart"), data.history || [], {
      x: (item) => new Date(item.time).getTime(), y: (item) => Number(item.total),
      xLabel: (value) => dateTime(new Date(value).toISOString()), yLabel: (value) => number(value),
      color: "#26d7c7", area: true,
    });
    $("#data-chart-empty").classList.toggle("hidden", (data.history || []).length > 0);
    renderCohortMetadata(data.current_cohort_metadata);
    renderQuarantine(data.quarantine);
    renderElectrostatic(data.electrostatic);
    renderThermalModels(data.thermal_models);
  }

  function renderCohortMetadata(metadataPayload) {
    const metadata = metadataPayload && typeof metadataPayload === "object" ? metadataPayload : {};
    const lamination = metadata.core_lamination_factor || {};
    const laminationRange = rangeSummary(lamination, "", 3);
    setText(
      "#cohort-lamination-factor",
      laminationRange.median === "—" ? "—" : `중앙값 ${laminationRange.median}`,
    );
    setText(
      "#cohort-lamination-detail",
      hasNumber(lamination.min) || hasNumber(lamination.max) || hasNumber(lamination.sample_count)
        ? laminationRange.detail
        : "표본 —",
    );

    const flux = metadata.winding_flux_linkage_readback || {};
    setText("#cohort-flux-availability", ratio(flux.available_rows, flux.cohort_rows, "개"));
    setText(
      "#cohort-flux-detail",
      hasNumber(flux.unavailable_rows) || hasNumber(flux.missing_rows)
        ? `미지원 ${count(flux.unavailable_rows)} · 누락 ${count(flux.missing_rows)}`
        : "상태 —",
    );
    const statuses = $("#cohort-flux-statuses");
    statuses.replaceChildren();
    const statusRows = Array.isArray(flux.statuses) ? flux.statuses : [];
    statusRows.forEach((item) => {
      statuses.append(element("span", "mini-chip", `${item?.status || "미지정"} ${count(item?.count)}`));
    });
  }

  function renderReasonList(container, reasonsPayload, emptyMessage) {
    const reasons = Array.isArray(reasonsPayload) ? reasonsPayload : [];
    container.replaceChildren();
    if (!reasons.length) {
      container.append(element("p", "empty-state compact-empty", emptyMessage));
      return;
    }
    reasons.forEach((item) => {
      const row = element("div", "reason-row");
      row.append(element("code", "", item?.reason || "미지정 사유"), element("strong", "", count(item?.count)));
      container.append(row);
    });
  }

  function renderQuarantine(payload) {
    const quarantine = payload && typeof payload === "object" ? payload : {};
    const current = quarantine.current && typeof quarantine.current === "object" ? quarantine.current : {};
    const legacy = quarantine.legacy && typeof quarantine.legacy === "object" ? quarantine.legacy : {};
    const hasCurrentReasons = Array.isArray(current.reasons);
    const currentReasons = hasCurrentReasons ? current.reasons : [];
    setText("#quarantine-current-title", `${current.label || "활성 코호트"} — ${count(current.rows)}`);
    setText("#quarantine-current-count", count(current.rows));
    setText("#quarantine-current-reason-count", hasCurrentReasons ? count(currentReasons.length, "건") : "—");
    renderReasonList($("#quarantine-current-reasons"), currentReasons, hasNumber(current.rows) && Number(current.rows) === 0
      ? "현재 코호트의 격리 행이 없습니다."
      : "현재 코호트 격리 정보를 사용할 수 없습니다.");
    setText("#quarantine-legacy-label", legacy.label || "레거시 코호트 노이즈");
    setText("#quarantine-legacy-count", count(legacy.rows));
    renderReasonList($("#quarantine-legacy-reasons"), legacy.reasons, hasNumber(legacy.rows) && Number(legacy.rows) === 0
      ? "레거시 격리 행이 없습니다."
      : "레거시 격리 정보를 사용할 수 없습니다.");
  }

  function renderRangeList(container, entries, unit) {
    container.replaceChildren();
    entries.forEach(([key, label, stats]) => {
      const source = stats && typeof stats === "object" ? stats : {};
      const unitKey = unit === "nF" ? "nF" : unit === "kHz" ? "kHz" : null;
      const values = {
        ...source,
        min: unitKey ? (source[`min_${unitKey}`] ?? source.min) : source.min,
        median: unitKey ? (source[`median_${unitKey}`] ?? source.median) : source.median,
        max: unitKey ? (source[`max_${unitKey}`] ?? source.max) : source.max,
      };
      const summary = rangeSummary(values, unit, 3);
      const row = element("div", "range-row");
      const identity = element("div", "range-identity");
      identity.append(element("b", "", label));
      identity.append(element("code", "", values.source_column || "source —"));
      const result = element("div", "range-values");
      result.append(element("strong", "", `중앙값 ${summary.median}`));
      result.append(element("small", "", summary.detail));
      row.dataset.metric = key;
      row.append(identity, result);
      container.append(row);
    });
  }

  function renderElectrostatic(payload) {
    const electrostatic = payload && typeof payload === "object" ? payload : {};
    const available = electrostatic.available === true;
    const active = electrostatic.active_cohort && typeof electrostatic.active_cohort === "object"
      ? electrostatic.active_cohort : {};
    const stateChip = $("#electrostatic-state");
    stateChip.className = `state-chip ${available ? "pass" : "unknown"}`;
    stateChip.textContent = available ? "STRICT" : active.available === false ? "데이터 없음" : "사용 불가";
    const cohortLabel = electrostatic.cohort_label || active.label || "활성 코호트 확인 중";
    setText(
      "#electrostatic-basis",
      active.available === false ? cohortLabel : `${cohortLabel} strict-full 기준`,
    );
    setText("#cap-stage-present", count(electrostatic.cap_stage_present_rows));
    setText("#cap-stage-absent", count(electrostatic.cap_stage_absent_rows));
    setText("#cap-stage-unknown", count(electrostatic.cap_stage_unknown_rows));
    setText("#electrostatic-cohort-rows", count(electrostatic.cohort_rows));
    const capacitance = electrostatic.capacitance || {};
    renderRangeList($("#capacitance-summary"), [
      ["tx_tx", "C_tx_tx", capacitance.tx_tx],
      ["rx_rx", "C_rx_rx", capacitance.rx_rx],
      ["tx_rx", "C_tx_rx", capacitance.tx_rx],
    ], "nF");
    const resonance = electrostatic.resonance || {};
    renderRangeList($("#resonance-summary"), [
      ["tx_self", "Tx self", resonance.tx_self],
      ["rx_self", "Rx self", resonance.rx_self],
      ["interwinding", "상호권선", resonance.interwinding],
    ], "kHz");
  }

  function thermalModelLabel(model) {
    const names = {
      isotropic_legacy: "등방성 레거시",
      anisotropic_wound_rule_of_mixtures_v1: "이방성 권선 혼합칙 v1",
    };
    return names[model] || model || "미지정 모델";
  }

  function thermalStat(label, statsPayload) {
    const stats = statsPayload && typeof statsPayload === "object" ? statsPayload : {};
    const summary = rangeSummary(stats, "W/m·K", 3);
    const row = element("div", "thermal-stat");
    row.append(element("span", "", label));
    const values = element("div");
    values.append(element("strong", "", summary.median), element("small", "", summary.detail));
    row.append(values);
    return row;
  }

  function renderThermalModels(payload) {
    const thermal = payload && typeof payload === "object" ? payload : {};
    const models = Array.isArray(thermal.models) ? thermal.models : [];
    const active = thermal.active_cohort && typeof thermal.active_cohort === "object"
      ? thermal.active_cohort : {};
    const cohortLabel = thermal.cohort_label || active.label || "활성 코호트 확인 중";
    setText("#thermal-model-basis", active.available === false ? cohortLabel : `${cohortLabel} 기준`);
    setText("#thermal-model-summary", thermal.available === true
      ? `${count(thermal.tagged_rows)} 태그`
      : active.available === false ? "데이터 없음" : "사용 불가");
    setText(
      "#thermal-model-missing",
      hasNumber(thermal.total_rows) || hasNumber(thermal.missing_rows)
        ? `전체 ${count(thermal.total_rows)} · 태그 ${count(thermal.tagged_rows)} · 누락 ${count(thermal.missing_rows)}`
        : "이 필드를 포함한 행이 없습니다.",
    );
    const list = $("#thermal-model-list");
    list.replaceChildren();
    if (!models.length) {
      list.append(element(
        "p",
        "empty-state compact-empty",
        active.available === false ? cohortLabel : "사용 가능한 열모델 태그 데이터가 없습니다.",
      ));
      return;
    }
    models.forEach((item) => {
      const card = element("article", "thermal-model-row");
      const heading = element("div", "thermal-model-heading");
      const identity = element("div");
      identity.append(element("strong", "", thermalModelLabel(item?.model)));
      identity.append(element("code", "", item?.model || "—"));
      const share = hasNumber(item?.percent) ? `${number(item.percent, 1)}%` : "—";
      heading.append(identity, element("b", "", `${count(item?.count)} · ${share}`));
      card.append(heading);
      const stats = element("div", "thermal-stat-list");
      stats.append(
        thermalStat("면내 k", item?.thermal_core_k_inplane),
        thermalStat("적층방향 k", item?.thermal_core_k_throughstack),
      );
      card.append(stats);
      list.append(card);
    });
  }

  function chartBounds(svg) {
    const viewBox = svg.viewBox.baseVal;
    return { width: viewBox.width || 760, height: viewBox.height || 260, left: 52, right: 18, top: 18, bottom: 34 };
  }

  function lineChart(svg, values, options) {
    svg.replaceChildren();
    if (!values.length) return;
    const box = chartBounds(svg);
    if (options.xTitle) box.bottom = Math.max(box.bottom, 48);
    const records = values
      .map((item, index) => ({ item, index, x: Number(options.x(item)), y: Number(options.y(item)) }))
      .filter((record) => Number.isFinite(record.x) && Number.isFinite(record.y))
      .sort((a, b) => a.x - b.x || a.index - b.index);
    if (!records.length) return;
    const xs = records.map((record) => record.x);
    const ys = records.map((record) => record.y);
    let xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    if (options.includeZero !== false) { yMin = Math.min(0, yMin); yMax = Math.max(0, yMax); }
    if (yMin === yMax) yMax = yMin + 1;
    const plotW = box.width - box.left - box.right;
    const plotH = box.height - box.top - box.bottom;
    const sx = (value) => box.left + (value - xMin) / (xMax - xMin) * plotW;
    const sy = (value) => box.top + (1 - (value - yMin) / (yMax - yMin)) * plotH;

    const defs = svgElement("defs");
    const gradient = svgElement("linearGradient", { id: `area-gradient-${svg.id}`, x1: 0, y1: 0, x2: 0, y2: 1 });
    gradient.append(svgElement("stop", { offset: "0%", "stop-color": options.color || "#26d7c7", "stop-opacity": .35 }));
    gradient.append(svgElement("stop", { offset: "100%", "stop-color": options.color || "#26d7c7", "stop-opacity": 0 }));
    defs.append(gradient); svg.append(defs);
    for (let i = 0; i <= 4; i += 1) {
      const yValue = yMin + (yMax - yMin) * i / 4;
      const y = sy(yValue);
      svg.append(svgElement("line", { x1: box.left, y1: y, x2: box.width - box.right, y2: y, class: "grid-line" }));
      const text = svgElement("text", { x: box.left - 8, y: y + 3, "text-anchor": "end" });
      text.textContent = options.yLabel(yValue); svg.append(text);
    }
    [0, .5, 1].forEach((ratio) => {
      const value = xMin + (xMax - xMin) * ratio;
      const text = svgElement("text", { x: sx(value), y: box.height - (options.xTitle ? 20 : 9), "text-anchor": ratio === 0 ? "start" : ratio === 1 ? "end" : "middle" });
      text.textContent = options.xLabel(value); svg.append(text);
    });
    if (options.xTitle) {
      const title = svgElement("text", { x: box.left + plotW / 2, y: box.height - 2, "text-anchor": "middle", class: "axis-title" });
      title.textContent = options.xTitle; svg.append(title);
    }
    const points = records.map((record) => ({ ...record, px: sx(record.x), py: sy(record.y) }));
    if (options.area && points.length) {
      const path = `M ${points[0].px} ${box.height - box.bottom} L ${points.map((point) => `${point.px} ${point.py}`).join(" L ")} L ${points.at(-1).px} ${box.height - box.bottom} Z`;
      svg.append(svgElement("path", { d: path, fill: `url(#area-gradient-${svg.id})` }));
    }
    svg.append(svgElement("path", { d: `M ${points.map((point) => `${point.px} ${point.py}`).join(" L ")}`, class: "data-line", stroke: options.color || "#26d7c7" }));

    const tooltip = svgElement("g", { class: "chart-tooltip", visibility: "hidden", "aria-hidden": "true" });
    const tooltipBox = svgElement("rect", { x: 0, y: 0, rx: 5, ry: 5 });
    const tooltipText = svgElement("text", { x: 9, y: 17 });
    tooltip.append(tooltipBox, tooltipText);
    const hideTooltip = () => tooltip.setAttribute("visibility", "hidden");
    const tooltipLines = (item) => {
      const raw = options.tooltip ? options.tooltip(item) : [];
      return (Array.isArray(raw) ? raw : [raw]).filter(Boolean).map(String);
    };
    const showTooltip = (point) => {
      const lines = tooltipLines(point.item);
      if (!lines.length) return;
      tooltipText.replaceChildren();
      lines.forEach((line, index) => {
        const span = svgElement("tspan", { x: 9, dy: index === 0 ? 0 : 15 });
        span.textContent = line; tooltipText.append(span);
      });
      const width = Math.min(270, Math.max(138, Math.max(...lines.map((line) => Array.from(line).length)) * 7.1 + 18));
      const height = lines.length * 15 + 10;
      let x = point.px + 10;
      if (x + width > box.width - 2) x = point.px - width - 10;
      let y = point.py - height - 10;
      if (y < 2) y = point.py + 10;
      tooltipBox.setAttribute("width", String(width));
      tooltipBox.setAttribute("height", String(height));
      tooltip.setAttribute("transform", `translate(${Math.max(2, x)} ${Math.min(box.height - height - 2, y)})`);
      tooltip.setAttribute("visibility", "visible");
    };

    points.forEach((point, index) => {
      if (!options.tooltip && points.length > 40 && index !== points.length - 1) return;
      const label = options.tooltip ? tooltipLines(point.item).join(" · ") : "";
      const circle = svgElement("circle", {
        cx: point.px, cy: point.py, r: options.tooltip ? 3.4 : 2.8,
        fill: options.color || "#26d7c7",
        class: options.tooltip ? "chart-data-point interactive" : "chart-data-point",
        ...(options.tooltip ? { tabindex: 0, role: "img", "aria-label": label } : {}),
      });
      if (options.tooltip) {
        circle.addEventListener("pointerenter", () => showTooltip(point));
        circle.addEventListener("pointerleave", hideTooltip);
        circle.addEventListener("focus", () => showTooltip(point));
        circle.addEventListener("blur", hideTooltip);
      }
      svg.append(circle);
    });
    if (options.tooltip) svg.append(tooltip);
  }

  function parityAxis(value) {
    const absolute = Math.abs(Number(value));
    if (absolute && (absolute >= 10000 || absolute < .01)) return Number(value).toExponential(2);
    return number(value, absolute < 10 ? 3 : absolute < 100 ? 2 : 1);
  }

  function paritySampleR2(points) {
    if (!Array.isArray(points) || points.length < 2) return null;
    const mean = points.reduce((sum, item) => sum + item.actual, 0) / points.length;
    const residual = points.reduce(
      (sum, item) => sum + ((item.actual - item.predicted) ** 2), 0,
    );
    const total = points.reduce(
      (sum, item) => sum + ((item.actual - mean) ** 2), 0,
    );
    return total > 0 ? 1 - residual / total : null;
  }

  function renderParityProvenance(model, payload = {}) {
    const note = $("#model-parity-provenance");
    const provenance = payload.parity_provenance || model.parity_provenance;
    const required = provenance?.required === true;
    note.classList.toggle("hidden", !required);
    note.classList.toggle("valid", required && provenance.valid === true);
    note.classList.toggle("invalid", required && provenance.valid !== true);
    if (!required) {
      note.textContent = "";
      return null;
    }
    if (provenance.valid !== true) {
      note.textContent = provenance.message
        || "legacy quantized labels / 교정 재학습 대기";
      return provenance;
    }
    const details = provenance.source_kind === "authenticated_overlay"
      ? [
        "SHA 인증 교정 OOF overlay",
        `checkpoint ${number(provenance.checkpoint)}`,
        `source ${provenance.overlay_path || "—"}`,
        `sha256 ${provenance.overlay_sha256 || "—"}`,
        `contract ${provenance.contract || "—"}`,
        `recovered ${number(provenance.recovered_row_count)}행`,
        `unique actual ${number(provenance.unique_actual_count)}개`,
      ]
      : [
        "LC 역산 교정 확인",
        `contract ${provenance.contract || "—"}`,
        `recovered ${number(provenance.recovered_row_count)}행`,
        `unique actual ${number(provenance.unique_actual_count)}개`,
      ];
    note.textContent = details.join(" · ");
    return provenance;
  }

  function renderParity(model, payload = {}) {
    const svg = $("#model-parity-chart");
    svg.replaceChildren();
    const provenance = renderParityProvenance(model, payload);
    const rawPairs = Array.isArray(payload.pairs)
      ? payload.pairs
      : Array.isArray(payload.parity) ? payload.parity : [];
    const points = rawPairs
      .map((item) => ({ actual: Number(item.actual), predicted: Number(item.predicted) }))
      .filter((item) => Number.isFinite(item.actual) && Number.isFinite(item.predicted));
    const empty = $("#model-parity-empty");
    empty.classList.toggle("hidden", points.length > 0);
    if (!points.length) {
      empty.textContent = payload.error
        || (provenance?.valid === false ? provenance.message : null)
        || "검증된 parity 데이터가 없습니다.";
      setText(
        "#model-parity-meta",
        provenance?.valid === false
          ? "invalid provenance"
          : model.evaluated ? `checkpoint ${model.checkpoint ?? "—"}` : "—",
      );
      return;
    }

    const values = points.flatMap((item) => [item.actual, item.predicted]);
    let low = Math.min(...values), high = Math.max(...values);
    if (low === high) {
      const spread = Math.max(1, Math.abs(low) * .05);
      low -= spread; high += spread;
    } else {
      const padding = (high - low) * .05;
      low -= padding; high += padding;
    }
    const box = chartBounds(svg);
    box.left = 70; box.bottom = 48;
    const plotW = box.width - box.left - box.right;
    const plotH = box.height - box.top - box.bottom;
    const scaleX = (value) => box.left + (value - low) / (high - low) * plotW;
    const scaleY = (value) => box.top + (1 - (value - low) / (high - low)) * plotH;

    for (let index = 0; index <= 4; index += 1) {
      const value = low + (high - low) * index / 4;
      const x = scaleX(value), y = scaleY(value);
      svg.append(svgElement("line", { x1: box.left, y1: y, x2: box.width - box.right, y2: y, class: "grid-line" }));
      svg.append(svgElement("line", { x1: x, y1: box.top, x2: x, y2: box.height - box.bottom, class: "grid-line" }));
      const yTick = svgElement("text", { x: box.left - 8, y: y + 3, "text-anchor": "end" });
      yTick.textContent = parityAxis(value); svg.append(yTick);
      const xTick = svgElement("text", { x, y: box.height - box.bottom + 17, "text-anchor": index === 0 ? "start" : index === 4 ? "end" : "middle" });
      xTick.textContent = parityAxis(value); svg.append(xTick);
    }
    svg.append(svgElement("line", {
      x1: scaleX(low), y1: scaleY(low), x2: scaleX(high), y2: scaleY(high), class: "identity-line",
    }));
    const xLabel = svgElement("text", { x: box.left + plotW / 2, y: box.height - 5, "text-anchor": "middle" });
    xLabel.textContent = `실제값${model.unit ? ` [${model.unit}]` : ""}`; svg.append(xLabel);
    const yLabel = svgElement("text", {
      x: 13, y: box.top + plotH / 2, transform: `rotate(-90 13 ${box.top + plotH / 2})`, "text-anchor": "middle",
    });
    yLabel.textContent = `OOF 예측값${model.unit ? ` [${model.unit}]` : ""}`; svg.append(yLabel);
    points.forEach((item) => {
      const point = svgElement("circle", {
        cx: scaleX(item.actual), cy: scaleY(item.predicted), r: points.length > 1000 ? 1.35 : 1.8,
        class: "parity-point",
      });
      const title = svgElement("title");
      title.textContent = `실제 ${compact(item.actual, model.unit)} · 예측 ${compact(item.predicted, model.unit)}`;
      point.append(title); svg.append(point);
    });
    const sampleCount = payload.sample_count ?? points.length;
    const totalCount = payload.n ?? model.n_used ?? sampleCount;
    const checkpoint = payload.checkpoint ?? model.checkpoint;
    const prefix = checkpoint == null ? "OOF" : `checkpoint ${checkpoint} OOF`;
    const uniqueActual = provenance?.valid === true
      ? ` · unique actual ${number(provenance.unique_actual_count)}` : "";
    const isOverlay = provenance?.source_kind === "authenticated_overlay";
    const r2 = isOverlay ? paritySampleR2(points) : model.r2;
    const r2Label = isOverlay ? "표시 표본 R²" : "R²";
    setText("#model-parity-meta", `${prefix} · ${number(sampleCount)}/${number(totalCount)}점 · ${r2Label} ${number(r2, 3)}${uniqueActual}`);
  }

  async function loadParity(model) {
    const inline = model.parity && typeof model.parity === "object" ? model.parity : null;
    if (inline) {
      renderParity(model, inline);
      return;
    }
    if (!model.parity_available) {
      renderParity(model, {});
      return;
    }
    const token = ++state.parityRequest;
    const cacheKey = [
      model.target, model.checkpoint, model.parity_checkpoint,
      model.evaluated_at, model.parity_sample_count,
      model.parity_provenance?.overlay_sha256,
    ].join(":");
    if (state.parityCache.has(cacheKey)) {
      renderParity(model, state.parityCache.get(cacheKey));
      return;
    }
    const empty = $("#model-parity-empty");
    empty.textContent = "Parity plot을 불러오는 중입니다.";
    empty.classList.remove("hidden");
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(model.target)}/parity`, {
        cache: "no-store", headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (payload.error) throw new Error(payload.error);
      state.parityCache.set(cacheKey, payload);
      if (token === state.parityRequest && state.selectedModel === model.target) renderParity(model, payload);
    } catch (error) {
      if (token === state.parityRequest && state.selectedModel === model.target) {
        renderParity(model, { error: `Parity 데이터를 불러오지 못했습니다: ${error.message}` });
      }
    }
  }

  function renderModels(payload) {
    const targetCount = Number(payload.target_count || 0);
    const checkpointEvaluated = (payload.models || [])
      .filter((model) => model.evaluation_kind === "checkpoint_cv").length;
    const summary = [];
    if (checkpointEvaluated) {
      const checkpoint = payload.latest_checkpoint == null ? "" : `${number(payload.latest_checkpoint)} 체크포인트 · `;
      summary.push(`${checkpoint}CV 평가 ${number(checkpointEvaluated)}/${number(targetCount)}`);
    }
    summary.push(`활성 모델 ${number(payload.trained_count)}/${number(targetCount)}`);
    if (payload.activation_minimum_strict_full_rows) {
      summary.push(`활성화 ${number(payload.current_data_count)}/${number(payload.activation_minimum_strict_full_rows)}`);
    }
    setText("#model-summary", summary.join(" · "));
    setText("#model-quality-note", payload.quality_note || "");
    const tbody = $("#model-table");
    tbody.replaceChildren();
    (payload.models || []).forEach((model) => {
      const row = element("tr"); row.dataset.clickable = "true"; row.dataset.target = model.target;
      const nameCell = element("td");
      const name = element("div", "model-name"); name.append(element("b", "", model.label), element("code", "", model.target)); nameCell.append(name);
      if (model.parity_provenance?.required === true) {
        name.append(element(
          "span",
          `parity-provenance-badge ${model.parity_provenance.valid === true ? "valid" : "invalid"}`,
          model.parity_provenance.source_kind === "authenticated_overlay"
            ? "authenticated OOF"
            : model.parity_provenance.valid === true ? "교정 parity" : "legacy labels",
        ));
      }
      row.append(nameCell);
      const evidence = model.trained
        ? `${number(model.n_train)}/${number(model.n_holdout)}`
        : model.evaluated ? `CV n=${number(model.n_used)}` : "—";
      row.append(element("td", "table-value", evidence));
      const r2 = element("td", "table-value", model.r2 == null ? "—" : number(model.r2, 3));
      if (model.delta_r2 != null) r2.append(element("span", model.delta_r2 >= 0 ? "delta-up" : "delta-down", ` ${model.delta_r2 >= 0 ? "▲" : "▼"}${number(Math.abs(model.delta_r2), 3)}`));
      row.append(r2);
      row.append(element("td", "table-value", compact(model.rmse, model.unit)));
      const mapeCell = element("td", "table-value", model.mape_pct == null ? "—" : `${number(model.mape_pct, 2)}%`);
      if (model.mape_n != null && Number(model.mape_excluded_zero_count || 0) > 0) {
        mapeCell.append(element(
          "small", "metric-sample",
          `0값 ${number(model.mape_excluded_zero_count)}개 제외 · n=${number(model.mape_n)}`,
        ));
      }
      row.append(mapeCell);
      const p90Cell = element("td", "table-value", model.p90_ape_pct == null ? "—" : `${number(model.p90_ape_pct, 2)}%`);
      if (model.mape_n != null) {
        p90Cell.title = `MAPE와 동일한 비영(非零) 실제값 ${number(model.mape_n)}개 기준`;
      }
      row.append(p90Cell);
      const statusCell = element("td"); statusCell.append(element("span", `state-chip ${model.status}`, labels[model.status] || model.status)); row.append(statusCell);
      row.addEventListener("click", () => selectModel(model));
      tbody.append(row);
    });
    let selected = (payload.models || []).find((model) => model.target === state.selectedModel);
    if (!selected) selected = (payload.models || []).find((model) => model.trained || model.evaluated) || payload.models?.[0];
    if (selected) selectModel(selected);
  }

  function historyMetricNumber(item, field) {
    if (item?.[field] == null || item[field] === "") return NaN;
    return Number(item[field]);
  }

  function historyMetricText(item, metric, model) {
    const value = historyMetricNumber(item, metric.field);
    if (!Number.isFinite(value)) return "—";
    if (metric.field === "rmse") return compact(value, model.unit);
    return `${number(value, metric.digits)}${metric.suffix || ""}`;
  }

  function historyPointTooltip(item, model) {
    return [
      `학습 데이터 ${number(item.n)}개`,
      `CV R² ${historyMetricText(item, historyMetrics.r2, model)}`,
      `CV MAPE ${historyMetricText(item, historyMetrics.mape_pct, model)}`,
      `CV RMSE ${historyMetricText(item, historyMetrics.rmse, model)}`,
      `CV P90 APE ${historyMetricText(item, historyMetrics.p90_ape_pct, model)}`,
      `평가 시각 ${dateTime(item.time)}`,
    ];
  }

  function renderModelHistory(model) {
    const metric = historyMetrics[state.historyMetric] || historyMetrics.r2;
    const selector = $("#model-history-metric");
    if (selector.value !== metric.field) selector.value = metric.field;
    setText("#model-history-title", `체크포인트 ${metric.label} 추세`);
    const history = (model.history || []).filter((item) => (
      Number.isFinite(Number(item.n))
      && Number(item.n) > 0
      && Number.isFinite(historyMetricNumber(item, metric.field))
    ));
    const empty = $("#model-chart-empty");
    empty.classList.toggle("hidden", history.length > 0);
    empty.textContent = history.length
      ? ""
      : `선택한 모델의 ${metric.label} 체크포인트 이력이 없습니다.`;
    const svg = $("#model-chart");
    svg.setAttribute("aria-label", `학습 데이터 수에 따른 ${model.label} ${metric.label} 추세`);
    lineChart(svg, history, {
      x: (item) => Number(item.n),
      y: (item) => historyMetricNumber(item, metric.field),
      xLabel: (value) => number(value),
      yLabel: (value) => metric.field === "rmse"
        ? compact(value, model.unit)
        : `${number(value, metric.field === "r2" ? 2 : 1)}${metric.suffix || ""}`,
      xTitle: "학습 데이터 수 [개]",
      tooltip: (item) => historyPointTooltip(item, model),
      color: metric.color,
      area: true,
    });
  }

  function selectModel(model) {
    state.selectedModel = model.target;
    state.selectedModelData = model;
    document.querySelectorAll("#model-table tr").forEach((row) => row.classList.toggle("selected", row.dataset.target === model.target));
    setText("#model-chart-title", model.label);
    if (model.trained) {
      setText("#model-evaluation-kind", "품질 게이트를 통과해 registry에 승격된 활성 모델");
      setText("#model-trained-at", model.trained_at ? `학습 ${dateTime(model.trained_at)}` : "활성 모델");
    } else if (model.status === "parity_overlay") {
      setText("#model-evaluation-kind", "모델 승격 전 SHA 인증 교정 OOF overlay · 실제 예측에는 사용되지 않음");
      setText("#model-trained-at", `OOF checkpoint ${model.parity_checkpoint ?? "—"} · metrics checkpoint ${model.checkpoint ?? "—"}`);
    } else if (model.evaluated) {
      setText("#model-evaluation-kind", "배포 전 체크포인트 교차검증(OOF) · 실제 예측에는 사용되지 않음");
      setText("#model-trained-at", `checkpoint ${model.checkpoint ?? "—"} · ${dateTime(model.evaluated_at)}`);
    } else {
      setText("#model-evaluation-kind", "아직 검증된 모델 평가 결과가 없습니다.");
      setText("#model-trained-at", "학습 전");
    }
    renderModelHistory(model);
    loadParity(model);
  }

  function nsgaGenerationSelection(payload) {
    const generations = Array.isArray(payload.pareto_generations)
      ? payload.pareto_generations.filter((item) => item && item.id)
      : [];
    const fallbackId = payload.selected_generation_id
      || generations.find((item) => item.active === true)?.id
      || generations[0]?.id
      || null;
    if (!state.selectedNsgaGenerationId
        || !generations.some((item) => (
          item.id === state.selectedNsgaGenerationId
          && item.selectable !== false
        ))) {
      const selectionChanged = state.selectedNsgaGenerationId !== fallbackId;
      state.selectedNsgaGenerationId = fallbackId;
      if (selectionChanged) {
        state.selectedCandidate = null;
        setLocalGuiButtons();
      }
    }
    const generation = generations.find(
      (item) => item.id === state.selectedNsgaGenerationId,
    ) || null;
    if (!generation || generation.active === true
        || generation.selectable === false) {
      return { generations, generation, view: payload, loading: false };
    }
    const cacheKey = nsgaGenerationCacheKey(generation);
    const cached = state.nsgaGenerationCache.get(cacheKey);
    if (nsgaGenerationDetailVerified(cached)) {
      return {
        generations,
        generation,
        view: {
          ...cached,
          pareto_generations: generations,
          selected_generation_id: generation.id,
          sealed_successor: payload.sealed_successor,
          sealed_successors: payload.sealed_successors,
        },
        loading: false,
      };
    }
    return {
      generations,
      generation,
      loading: true,
      view: {
        schema_version: payload.schema_version,
        available: true,
        status: "waiting",
        al_stage: "CONDITION HISTORY",
        candidate_count: 0,
        display_candidate_count: 0,
        valid_candidate_count: 0,
        candidates: [],
        summary: {
          candidate_count: 0,
          display_candidate_count: 0,
          valid_candidate_count: 0,
          min_volume_L: null,
          min_loss_W: null,
        },
        constraint_version: generation.constraint_version,
        constraints: generation.hard_spec || {},
        selected_generation_id: generation.id,
        generation,
        note: "선택한 조건 세대의 인증된 Pareto 결과를 불러오는 중입니다.",
        tier1_feedback_search: {},
      },
    };
  }

  function nsgaGenerationCacheKey(generation) {
    return [
      generation?.id || "",
      generation?.cohort_id || "",
      generation?.updated_at || "",
      generation?.candidate_count ?? "",
      generation?.display_candidate_count ?? "",
      generation?.near_feasible_count ?? "",
      generation?.state || "",
    ].join("|");
  }

  function nsgaGenerationDetailVerified(detail) {
    return detail?.available === true && detail?.integrity_verified === true;
  }

  function beginNsgaGenerationRequest(cache, requests, generation) {
    const generationId = String(generation?.id || "");
    const cacheKey = nsgaGenerationCacheKey(generation);
    if (!generationId || requests.has(generationId)
        || nsgaGenerationDetailVerified(cache.get(cacheKey))) return null;
    for (const key of cache.keys()) {
      if (key.startsWith(`${generationId}|`)) cache.delete(key);
    }
    requests.add(generationId);
    return { cacheKey, generationId };
  }

  function settleNsgaGenerationRequest(cache, requests, request, detail) {
    if (!request) return false;
    requests.delete(request.generationId);
    if (nsgaGenerationDetailVerified(detail)) {
      cache.set(request.cacheKey, detail);
      return true;
    }
    cache.delete(request.cacheKey);
    return false;
  }

  async function loadNsgaGeneration(generation) {
    if (!generation || generation.active === true
        || generation.selectable === false) return;
    const request = beginNsgaGenerationRequest(
      state.nsgaGenerationCache,
      state.nsgaGenerationRequests,
      generation,
    );
    if (!request) return;
    try {
      const response = await fetch(
        generation.candidate_endpoint
          || `/api/nsga2/generations/${encodeURIComponent(generation.id)}`,
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const detail = await response.json();
      if (!settleNsgaGenerationRequest(
        state.nsgaGenerationCache,
        state.nsgaGenerationRequests,
        request,
        detail,
      )) {
        throw new Error(
          detail?.error || detail?.warning
          || "archived generation detail is unavailable or failed integrity verification",
        );
      }
      if (state.selectedNsgaGenerationId === generation.id
          && state.nsgaPayload) {
        renderNsga(state.nsgaPayload, state.nsgaProgress);
      }
    } catch (error) {
      settleNsgaGenerationRequest(
        state.nsgaGenerationCache,
        state.nsgaGenerationRequests,
        request,
        null,
      );
      if (state.selectedNsgaGenerationId === generation.id) {
        const stateNode = $("#nsga-generation-state");
        stateNode.className = "state-chip fail";
        stateNode.textContent = "로드 실패";
        const summary = $("#nsga-generation-summary");
        summary.className = "nsga-generation-summary error";
        summary.textContent = `조건 세대 로드 실패: ${error.message}`;
      }
    }
  }

  function renderNsgaGenerationSelector(rootPayload, selection) {
    const container = $("#nsga-generation-selector");
    container.replaceChildren();
    selection.generations.forEach((generation) => {
      const button = element(
        "button",
        `nsga-generation-button${generation.id === selection.generation?.id ? " active" : ""}`
          + `${generation.state === "incomplete" ? " incomplete" : ""}`,
      );
      button.type = "button";
      button.role = "tab";
      button.disabled = generation.selectable === false;
      if (button.disabled) {
        button.title = generation.warning || "완료된 인증 결과가 없는 조건 세대입니다.";
      }
      button.dataset.generationId = generation.id;
      button.setAttribute(
        "aria-selected",
        generation.id === selection.generation?.id ? "true" : "false",
      );
      const fullCount = Number(generation.candidate_count || 0);
      const previewCount = Number(generation.display_candidate_count || 0);
      const preview = fullCount !== previewCount
        ? ` · preview ${number(previewCount)}` : "";
      const sourceCount = Number(generation.source_count || 0);
      const sourceSuffix = sourceCount > 1
        ? ` · sources ${number(sourceCount)}` : "";
      const nearCount = Number(generation.near_feasible_count || 0);
      const nearSuffix = nearCount > 0
        ? ` · near ${number(nearCount)}` : "";
      const refusedCount = Number(generation.refused_terminal_count || 0);
      const refusalSuffix = refusedCount > 0
        ? ` · refused ${number(refusedCount)}` : "";
      button.append(
        element("strong", "", generation.label || generation.constraint_version),
        element(
          "small",
          "",
          `Pareto ${number(fullCount)}${preview}${nearSuffix}${refusalSuffix} · ${generation.active ? "active" : generation.state || "archived"}${sourceSuffix}`,
        ),
      );
      button.addEventListener("click", () => {
        if (generation.selectable === false) return;
        if (state.selectedNsgaGenerationId === generation.id) {
          const cacheKey = nsgaGenerationCacheKey(generation);
          if (generation.active === true
              || state.nsgaGenerationRequests.has(String(generation.id))
              || nsgaGenerationDetailVerified(
                state.nsgaGenerationCache.get(cacheKey),
              )) return;
          renderNsga(rootPayload, state.nsgaProgress);
          return;
        }
        state.selectedNsgaGenerationId = generation.id;
        state.selectedCandidate = null;
        setLocalGuiButtons();
        renderNsga(rootPayload, state.nsgaProgress);
      });
      container.append(button);
    });
    const generation = selection.generation;
    const historicalLoaded = Boolean(
      generation && generation.active !== true && !selection.loading,
    );
    const detailVerified = generation?.active === true || Boolean(
      historicalLoaded
      && selection.view?.available === true
      && selection.view?.integrity_verified === true,
    );
    const detailFailed = historicalLoaded && !detailVerified;
    const noGeneration = !generation;
    const stateNode = $("#nsga-generation-state");
    stateNode.className = `state-chip ${selection.loading || noGeneration ? "unknown" : detailFailed ? "fail" : generation?.state === "incomplete" ? "attention" : "pass"}`;
    stateNode.textContent = selection.loading
      ? "Pareto 인증 중"
      : noGeneration ? "조건 없음"
        : detailFailed ? "무결성 인증 실패"
        : generation?.active ? "현재 조건"
          : generation?.state === "incomplete" ? "미완료 이력" : "이전 조건 · 인증 완료";
    const summary = $("#nsga-generation-summary");
    const detailWarnings = Array.isArray(selection.view?.warnings)
      ? selection.view.warnings.filter(Boolean).join(" · ") : "";
    const detailWarning = selection.view?.warning || detailWarnings;
    summary.className = `nsga-generation-summary${detailFailed ? " error" : ""}`;
    summary.textContent = generation
      ? detailFailed
        ? detailWarning || "과거 조건 세대의 terminal/result/Pareto 무결성 인증에 실패했습니다."
        : generation.warning || `${generation.constraint_version} · 전체 Pareto ${number(generation.candidate_count || 0)}개 · near ${number(generation.near_feasible_count || 0)}개`
      : "표시할 인증된 조건 세대가 없습니다.";
    if (selection.loading && generation) loadNsgaGeneration(generation);
  }

  function tier1SourceProgress(search = {}) {
    const sources = Array.isArray(search.sources)
      ? search.sources.filter((source) => source?.accepted === true)
      : [];
    if (sources.length < 2) return "";
    const details = sources.map((source) => {
      const label = source.role === "primary"
        ? "main" : source.label || source.cohort_id || "aux";
      return `${label} ${number(source.completed_count || 0)}/${number(source.search_count || 0)}`;
    });
    return ` · sources ${details.join(" + ")}`;
  }

  function nearViolationLabel(violation = {}) {
    const key = String(violation.constraint || "unknown");
    const amount = Number(violation.amount);
    if (!Number.isFinite(amount) || amount <= 0) return null;
    const temperatureTargets = {
      T_max_Tx: "Tx",
      T_max_Rx_main: "Rx-main",
      T_max_Rx_side: "Rx-side",
      T_max_core: "core",
      Tprobe_Tx_leeward_max: "Tx probe",
      Tprobe_Rx_main_leeward_max: "Rx-main probe",
      Tprobe_Rx_side_leeward_max: "Rx-side probe",
      Tprobe_core_center_max: "core center",
      Tprobe_core_center_leg_max: "core center-leg",
      Tprobe_core_side_leg_max: "core side-leg",
      Tprobe_core_top_yoke_max: "core yoke",
    };
    if (key.startsWith("temperature_robust_limit:")) {
      const target = key.split(":", 2)[1] || "temperature";
      return `${temperatureTargets[target] || target} +${number(amount, 2)} °C`;
    }
    const formats = {
      Llt_robust_band: ["Llt robust", "µH", 3],
      Llt_ensemble_disagreement: ["Llt 모델편차", "µH", 3],
      half_magnetizing_resonance_minimum: ["공진 부족", "kHz", 3, 1e-3],
      half_magnetizing_resonance_maximum: ["공진 초과", "kHz", 3, 1e-3],
      analytical_flux_density_limit: ["설계 B", "T", 3],
      exterior_width_limit: ["폭 W", "mm", 1],
      exterior_length_limit: ["길이 L", "mm", 1],
      exterior_height_limit: ["높이 H", "mm", 1],
      minimum_physical_insulation: ["절연간격 부족", "mm", 2],
      core_group_manufacturability_limit: ["코어 조 수", "", 2],
      strict_full_density_support: ["학습밀도", "", 3],
      decoded_space_shrink: ["학습범위", "", 3],
    };
    const [label, unit, digits, scale = 1] = formats[key]
      || [key, "", 3, 1];
    return `${label} +${number(amount * scale, digits)}${unit ? ` ${unit}` : ""}`;
  }

  function candidateDialogFacts(candidate = {}) {
    const violations = Array.isArray(candidate.violations)
      ? candidate.violations.map(nearViolationLabel).filter(Boolean)
      : [];
    return {
      isReadOnlyNear: (
        candidate.valid_pareto === false
        && candidate.gui_launch_eligible === false
      ),
      predictedLeakage: candidateNumber(
        candidate,
        "pred_Llt_phys",
        "pred_Llt_phys_uH",
        "pred_leakage_inductance_uH",
      ),
      suppliedMaxTemperature: candidateNumber(
        candidate,
        "pred_max_temperature_C",
        "robust_max_temperature_C",
        "max_temperature_C",
        "T_max_C",
      ),
      violationLabels: violations,
    };
  }

  function renderNsgaNearPreview(payload = {}, tier1 = {}) {
    const panel = $("#nsga-near-panel");
    if (!panel) return;
    const preview = Array.isArray(payload.near_feasible_preview)
      ? payload.near_feasible_preview : [];
    const rows = preview.filter((item) => (
      item && item.valid_pareto === false
      && item.production_eligible === false
      && item.fea_submission_approved === false
      && item.gui_launch_eligible === false
    ));
    const total = Number(
      tier1.near_feasible_count
      ?? payload.near_feasible_count
      ?? rows.length
    );
    const visible = rows.length > 0 || (Number.isFinite(total) && total > 0);
    panel.classList.toggle("hidden", !visible);
    if (!visible) return;

    const stateNode = $("#nsga-near-state");
    stateNode.className = "state-chip attention";
    stateNode.textContent = `유효 Pareto 아님 · near ${number(total)}`;
    setText(
      "#nsga-near-summary",
      `인증된 near-feasible ${number(total)}개 중 preview ${number(rows.length)}개 · 아직 hard constraint 미통과 · GUI/FEA 자동 제출 불가`,
    );
    const tbody = $("#nsga-near-table");
    tbody.replaceChildren();
    rows.slice(0, 20).forEach((candidate) => {
      const row = element("tr");
      row.dataset.clickable = "true";
      row.tabIndex = 0;
      const source = element("td");
      const sourceBox = element("div", "nsga-near-source");
      sourceBox.append(
        element(
          "b",
          "",
          `${candidate.source_label || candidate.source_role || "main"} · seed ${number(candidate.seed)}`,
        ),
        element("code", "", `task ${number(candidate.task_id)}`),
      );
      source.append(sourceBox);
      row.append(source);
      const dimensions = [
        candidate.size_W_mm,
        candidate.size_L_mm,
        candidate.size_H_mm,
      ];
      row.append(element(
        "td",
        "table-value",
        dimensions.every((value) => Number.isFinite(Number(value)))
          ? `${number(dimensions[0], 1)} × ${number(dimensions[1], 1)} × ${number(dimensions[2], 1)} mm`
          : "—",
      ));
      row.append(element("td", "table-value", compact(candidate.volume_L, "L")));
      const lltLower = Number(candidate.Llt_robust_lower_uH);
      const lltUpper = Number(candidate.Llt_robust_upper_uH);
      row.append(element(
        "td",
        "table-value",
        Number.isFinite(lltLower) && Number.isFinite(lltUpper)
          ? `${number(lltLower, 3)}–${number(lltUpper, 3)} µH`
          : compact(candidate.pred_Llt_phys_uH, "µH"),
      ));
      row.append(element(
        "td",
        "table-value",
        Number.isFinite(Number(candidate.f_res_min_screen_Hz))
          ? compact(Number(candidate.f_res_min_screen_Hz) / 1000, "kHz")
          : "—",
      ));
      row.append(element(
        "td",
        "table-value",
        compact(candidate.robust_max_temperature_C, "°C"),
      ));
      const violations = Array.isArray(candidate.violations)
        ? candidate.violations.map(nearViolationLabel).filter(Boolean)
        : [];
      row.append(element(
        "td",
        "nsga-near-violations",
        violations.length ? violations.slice(0, 4).join(" · ") : "미통과 제약 상세 없음",
      ));
      row.addEventListener("click", () => showCandidate(candidate));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter") showCandidate(candidate);
      });
      tbody.append(row);
    });
  }

  function renderNsga(rootPayload, progressPayload = null) {
    state.nsgaPayload = rootPayload;
    const selection = nsgaGenerationSelection(rootPayload);
    renderNsgaGenerationSelector(rootPayload, selection);
    const viewingActiveGeneration = !selection.generation
      || selection.generation.active === true;
    const payload = viewingActiveGeneration
      ? nsgaAuthorityView(selection.view)
      : selection.view;
    renderSealedSuccessors(
      rootPayload.sealed_successors || {},
      rootPayload.sealed_successor || {},
    );
    setText(
      "#nsga-status",
      selection.loading
        ? "이전 조건 Pareto 인증 중"
        : viewingActiveGeneration
          ? payload.available
            ? `${payload.status === "running" ? "실행 중" : "완료"} · AL ${payload.al_stage || "—"}`
            : "실행 전"
          : payload.available ? "이전 조건 · 인증 완료" : "이전 조건 · 유효 Pareto 없음",
    );
    setText("#nsga-round", payload.round == null ? "—" : `#${String(payload.round).padStart(2, "0")}`);
    const displayedCandidates = Number(
      payload.display_candidate_count ?? payload.candidate_count ?? 0
    );
    const validCandidates = Number(
      payload.valid_candidate_count ?? payload.summary?.valid_candidate_count ?? 0
    );
    const candidateLabel = payload.result_scope === "historical"
      ? `현재 유효 ${number(validCandidates)}개 · 과거 진단 ${number(displayedCandidates)}개`
      : displayedCandidates === validCandidates
        ? `유효 ${number(validCandidates)}개`
        : `유효 ${number(validCandidates)}개 · 탐색 ${number(displayedCandidates)}개`;
    setText(
      "#nsga-count",
      payload.available || selection.generation ? candidateLabel : "—",
    );
    const tier1 = nsgaAuthoritySearch(payload);
    if (payload.available && tier1.available && tier1.candidate_preview_truncated === true) {
      setText(
        "#nsga-count",
        `${candidateLabel} · Tier-1 전체 Pareto ${number(tier1.feasible_pareto_count || 0)}개 (상세 preview ${number(tier1.candidate_preview_count || 0)}개)`,
      );
    }
    const progress = viewingActiveGeneration
      && progressPayload?.integrity_verified === true
      ? progressPayload
      : tier1;
    const tier1Preview = tier1.candidate_preview_truncated === true
      ? ` (preview ${number(tier1.candidate_preview_count || 0)}/${number(tier1.candidate_preview_limit || 0)})`
      : "";
    setText(
      "#nsga-tier1-progress",
      tier1.available
        ? `${number(tier1.completed_count || 0)}/${number(tier1.search_count || 0)} 완료 · ${number(tier1.running_count || 0)} 실행 · feasible ${number(tier1.feasible_pareto_count || 0)}${tier1Preview}`
          + tier1SourceProgress(tier1)
        : "—",
    );
    if (progress.available) {
      const progressPreview = progress.candidate_preview_truncated === true
        ? ` (preview ${number(progress.candidate_preview_count || 0)}/${number(progress.candidate_preview_limit || 0)})`
        : "";
      setText(
        "#nsga-tier1-progress",
        `terminal ${number(progress.terminal_results_verified ?? progress.completed_count ?? 0)}`
          + ` (failed ${number(progress.failed_terminal_results_verified ?? 0)})`
          + ` · active+queued ${number(progress.active_plus_queued ?? 0)}`
          + ` (running ${number(progress.running_count || 0)}, queued ${number(progress.queued_count || 0)}, attaching ${number(progress.attaching_count || 0)})`
          + ` · Pareto ${number(progress.feasible_pareto_count || 0)}${progressPreview}`
          + ` · near ${number(progress.near_feasible_count || 0)}`
          + tier1SourceProgress(progress),
      );
    }
    const activeConstraints = payload.constraints || tier1.constraints || {};
    const constraintVersion = payload.constraint_version || tier1.constraint_version || "unavailable";
    const dimensions = activeConstraints.max_dimensions_mm || [
      activeConstraints.max_width_mm ?? activeConstraints.size_W_max_mm,
      activeConstraints.max_length_mm ?? activeConstraints.size_L_max_mm,
      activeConstraints.max_height_mm ?? activeConstraints.size_H_max_mm,
    ];
    const dimensionsLabel = Array.isArray(dimensions) && dimensions.every((value) => value != null)
      ? `${dimensions.join(" × ")} mm` : "unavailable";
    const resonanceMinimum = activeConstraints.min_resonance_hz
      ?? activeConstraints.resonance_min_hz
      ?? activeConstraints.resonance_min_Hz;
    const resonanceMaximum = activeConstraints.max_resonance_hz
      ?? activeConstraints.resonance_max_hz
      ?? activeConstraints.resonance_max_Hz;
    const resonanceLabel = resonanceMinimum != null && resonanceMaximum != null
      ? `${compact(resonanceMinimum / 1000, "kHz")} ≤ f < ${compact(resonanceMaximum / 1000, "kHz")}`
      : resonanceMaximum != null
        ? `< ${compact(resonanceMaximum / 1000, "kHz")}`
        : resonanceMinimum != null
          ? `≥ ${compact(resonanceMinimum / 1000, "kHz")}`
          : "unavailable";
    const temperature = activeConstraints.robust_temperature_max_C
      ?? activeConstraints.max_temperature_C
      ?? activeConstraints.temperature_upper_C
      ?? activeConstraints.T_limit_C;
    const coreGroups = activeConstraints.n_core_group_max;
    const primaryThickness = activeConstraints.primary_conductor_thickness_mm;
    const magnetizingFactor = activeConstraints.magnetizing_inductance_factor;
    setText(
      "#nsga-constraint-contract",
      `${viewingActiveGeneration ? "Active" : "Archived"} constraint ${constraintVersion} · max ${dimensionsLabel}`
        + ` · resonance ${resonanceLabel}`
        + ` · robust temperature ≤ ${temperature == null ? "unavailable" : compact(temperature, "°C")}`,
    );
    const constraintContract = $("#nsga-constraint-contract");
    constraintContract.textContent += ` · core groups <= ${coreGroups ?? "unavailable"}`
      + ` · primary ${primaryThickness == null ? "unavailable" : compact(primaryThickness, "mm")}`
      + ` · Lm factor ${magnetizingFactor ?? "unavailable"}`;
    setText("#nsga-min-volume", compact(payload.summary?.min_volume_L, "L"));
    setText("#nsga-min-loss", compact(payload.summary?.min_loss_W, "W"));
    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const nearCandidates = Array.isArray(payload.near_feasible_preview)
      ? payload.near_feasible_preview.filter((candidate) => (
        candidate && candidate.valid_pareto === false
        && candidate.production_eligible === false
        && candidate.fea_submission_approved === false
        && candidate.gui_launch_eligible === false
      ))
      : [];
    const comparison = payload.comparison;
    const comparisonSuffix = comparison?.min_volume_change_L == null
      ? ""
      : ` · 이전 대비 최소체적 ${comparison.min_volume_change_L > 0 ? "+" : ""}${number(comparison.min_volume_change_L, 1)} L`;
    const nearTotal = Number(
      tier1.near_feasible_count
      ?? payload.near_feasible_count
      ?? nearCandidates.length
    );
    setText(
      "#nsga-comparison",
      nearCandidates.length > 0
        ? `유효 Pareto ${number(candidates.length)}개 · 근접 후보 ${number(nearTotal)}개 (주황색)${comparisonSuffix}`
        : comparisonSuffix
          ? comparisonSuffix.slice(3)
          : "이전 round 비교 없음",
    );
    setText("#nsga-note", payload.note || "");
    renderNsgaNearPreview(payload, tier1);
    renderScatter(candidates, nearCandidates);
    const emptyState = $("#nsga-empty");
    emptyState.classList.toggle("hidden", candidates.length > 0);
    emptyState.textContent = (
      candidates.length === 0
      && Array.isArray(payload.near_feasible_preview)
      && payload.near_feasible_preview.length > 0
    )
      ? "유효 Pareto는 아직 0개입니다. 아래에 제약 미통과 근접 후보를 별도로 표시합니다."
      : "아직 NSGA-II 결과가 없습니다.";
    const tbody = $("#candidate-table"); tbody.replaceChildren();
    candidates.slice(0, 20).forEach((candidate) => {
      const row = element("tr", candidate.is_min_volume ? "minimum" : ""); row.dataset.clickable = "true";
      row.append(element("td", "candidate-id", candidate.id));
      row.append(element("td", "table-value", compact(candidate.volume_L, "L")));
      row.append(element("td", "table-value", compact(candidate.total_loss_W, "W")));
      row.append(element("td", "table-value", compact(candidate.pred_Llt_phys, "µH")));
      row.append(element("td", "table-value", compact(candidate.B_design_analytic_T, "T")));
      const status = element("td"); status.append(element("span", `state-chip ${candidate.spec_status}`, labels[candidate.spec_status])); row.append(status);
      row.addEventListener("click", () => showCandidate(candidate)); tbody.append(row);
    });
  }

  function appendDefinitionItems(selector, items) {
    const container = $(selector);
    container.replaceChildren();
    items.forEach(([label, value]) => {
      const item = element("div");
      item.append(element("dt", "", label), element("dd", "", value == null ? "—" : String(value)));
      container.append(item);
    });
  }

  function sealedSuccessorBox(candidate = {}) {
    const box = Array.isArray(candidate.exterior_box_mm) ? candidate.exterior_box_mm : [];
    return box.length === 3 && box.every((value) => Number.isFinite(Number(value)))
      ? box.map((value) => compact(value, "mm")).join(" × ")
      : "—";
  }

  function renderSealedSuccessor(payload = {}) {
    const panel = $("#sealed-successor-panel");
    const configured = payload.configured === true;
    const verified = payload.available === true && payload.integrity_verified === true;
    panel.classList.toggle("hidden", !configured);
    state.sealedSuccessor = verified ? payload : null;
    if (!configured) return;

    const badge = $("#sealed-successor-integrity");
    badge.className = `state-chip ${verified ? "pass" : "fail"}`;
    badge.textContent = verified ? "INTEGRITY PASS" : "INTEGRITY REJECTED";
    const candidate = verified ? payload.candidate || {} : {};
    const resonance = candidate.resonance || {};
    const llt = candidate.Llt_robust || {};
    const validation = verified && payload.standard_fea_validation
      && typeof payload.standard_fea_validation === "object"
      ? payload.standard_fea_validation
      : {};
    const validationVerified = validation.available === true
      && validation.integrity_verified === true;
    const validationStatus = validationVerified
      ? validation.task_status || validation.runtime_state || "unknown"
      : validation.configured === true ? "rejected" : "not configured";
    setText(
      "#sealed-successor-title",
      `${payload.display_label || "Sealed successor"} / Standard-FEA ${validationStatus}`,
    );
    setText("#sealed-successor-id", verified ? payload.evidence_id : "rejected");
    setText("#sealed-successor-box", verified ? sealedSuccessorBox(candidate) : "—");
    setText("#sealed-successor-resonance", verified ? compact(Number(resonance.screen_Hz) / 1000, "kHz") : "—");
    setText("#sealed-successor-temperature", verified ? compact(candidate.worst_temperature_C, "°C") : "—");
    setText("#sealed-successor-llt", verified ? compact(llt.G_uH, "µH") : "—");
    setText(
      "#sealed-successor-authority",
      verified ? "production=false · sealed FEA=false" : "fail-closed",
    );
    const validationNode = $("#sealed-successor-validation");
    const validationTaskId = validation.task && validation.task.id != null
      ? ` #${validation.task.id}`
      : "";
    validationNode.textContent = `${validationStatus}${validationTaskId}`;
    validationNode.className = `standard-fea-overlay ${String(validationStatus).toLowerCase()}`;
    setText(
      "#sealed-successor-description",
      verified
        ? "독립 replay까지 해시 검증된 successor입니다. canonical Pareto/terminal 개수에는 포함하지 않으며 FEA 검증 전까지 읽기 전용입니다."
        : "Sealed successor 증거 무결성 검증에 실패해 설계값 노출을 차단했습니다.",
    );
    const warning = $("#sealed-successor-warning");
    const warningText = payload.warning || validation.warning || "";
    warning.textContent = warningText;
    warning.classList.toggle("hidden", !warningText);
    $("#sealed-successor-detail-button").disabled = !verified;
    const dialog = $("#sealed-successor-dialog");
    if (!verified && dialog.open) dialog.close();
    if (verified && dialog.open) populateSealedSuccessorDialog(payload);
  }

  function renderSealedSuccessors(collection = {}, fallback = {}) {
    const verifiedCollection = collection.available === true
      && collection.integrity_verified === true
      && Array.isArray(collection.candidates)
      && collection.candidates.length === 2;
    const candidates = verifiedCollection
      ? collection.candidates.filter((candidate) => (
        candidate?.source_kind === "sealed_successor_evidence"
        && candidate?.terminal_result === false
        && candidate?.canonical_candidate === false
        && candidate?.pareto === false
        && candidate?.production === false
        && candidate?.fea === false
        && candidate?.fea_submission_performed === false
      ))
      : [];
    const grid = $("#sealed-successor-candidates");
    grid.replaceChildren();
    grid.classList.toggle("hidden", candidates.length !== 2);
    state.sealedSuccessors = candidates;

    if (candidates.length !== 2) {
      const rejected = collection.configured === true && collection.available !== true;
      renderSealedSuccessor(
        fallback.configured === true
          ? fallback
          : rejected
            ? { ...collection, configured: true, available: false, integrity_verified: false }
            : fallback,
      );
      return;
    }

    let selected = candidates.find((candidate) => (
      candidate.evidence_id === state.sealedSuccessorId
    )) || candidates[0];
    const select = (candidate) => {
      selected = candidate;
      state.sealedSuccessorId = candidate.evidence_id;
      renderSealedSuccessor(candidate);
      grid.querySelectorAll(".sealed-successor-candidate-card").forEach((card) => {
        card.classList.toggle("active", card.dataset.evidenceId === candidate.evidence_id);
      });
      const selectedValidation = candidate.standard_fea_validation || {};
      const selectedValidationStatus = selectedValidation.available === true
        && selectedValidation.integrity_verified === true
        ? selectedValidation.task_status || selectedValidation.runtime_state || "unknown"
        : selectedValidation.configured === true ? "rejected" : "not configured";
      setText(
        "#sealed-successor-title",
        `Sealed successor evidence · ${candidates.length} candidates · Standard-FEA ${selectedValidationStatus}`,
      );
      const badge = $("#sealed-successor-integrity");
      badge.className = "state-chip pass";
      badge.textContent = `INTEGRITY PASS · ${candidates.length}/${candidates.length}`;
      setText(
        "#sealed-successor-description",
        "ValidateOnly로 봉인된 exact와 interior 증거입니다. 두 후보 모두 canonical Pareto/terminal/production/FEA-completed 집계와 분리되어 있으며 현재 선택한 카드만 아래에 표시합니다.",
      );
      const warning = $("#sealed-successor-warning");
      const warningText = collection.warning || selectedValidation.warning || "";
      warning.textContent = warningText;
      warning.classList.toggle("hidden", !warningText);
    };
    candidates.forEach((candidate) => {
      const card = element("button", "sealed-successor-candidate-card");
      card.type = "button";
      card.dataset.evidenceId = candidate.evidence_id;
      const design = candidate.candidate || {};
      const validation = candidate.standard_fea_validation || {};
      const validationStatus = validation.available === true
        && validation.integrity_verified === true
        ? validation.task_status || validation.runtime_state || "unknown"
        : validation.configured === true ? "rejected" : "not configured";
      const validationTaskId = validation.task && validation.task.id != null
        ? ` #${validation.task.id}`
        : "";
      const validationLine = element(
        "small",
        `standard-fea-overlay ${String(validationStatus).toLowerCase()}`,
        `Standard-FEA: ${validationStatus}${validationTaskId}`,
      );
      card.append(
        element("strong", "", candidate.display_label || candidate.evidence_id),
        element("span", "", `${candidate.evidence_id} · ${String(design.candidate_sha256 || "").slice(0, 12)}`),
        element("small", "", "sealed: terminal=false · Pareto=false · production=false · FEA=false"),
        validationLine,
      );
      card.addEventListener("click", () => select(candidate));
      grid.append(card);
    });
    select(selected);
  }

  function populateSealedSuccessorDialog(payload) {
    const candidate = payload.candidate || {};
    const integrity = payload.integrity || {};
    const model = payload.model_identity || {};
    const hard = payload.hard_spec || {};
    const llt = candidate.Llt_robust || {};
    const resonance = candidate.resonance || {};
    const validation = payload.standard_fea_validation || {};
    const validationTask = validation.task || {};
    const resultIdentity = validation.result_identity || {};
    const validationIntegrity = validation.integrity || {};
    const measurements = validation.measurements || {};
    const actualGates = validation.actual_gates || {};
    const actual = actualGates.actual || {};
    const validationVerified = validation.available === true
      && validation.integrity_verified === true;
    const validationStatus = validationVerified
      ? validation.task_status || validation.runtime_state || "unknown"
      : validation.configured === true ? "rejected" : "not configured";
    setText(
      "#sealed-successor-dialog-title",
      `${payload.evidence_id || "Sealed successor"} / sealed FEA false / Standard-FEA ${validationStatus}`,
    );
    const summary = $("#sealed-successor-dialog-summary");
    summary.replaceChildren();
    [
      ["외형 W×L×H", sealedSuccessorBox(candidate), "geometry-card"],
      ["체적", compact(candidate.objective_volume_L, "L")],
      ["총손실", compact(candidate.objective_total_loss_W, "W")],
      ["0.5×Lm 공진", compact(Number(resonance.screen_Hz) / 1000, "kHz")],
      ["Llt robust G", compact(llt.G_uH, "µH")],
      ["All11 robust 최고온도", compact(candidate.worst_temperature_C, "°C")],
      ["Production", "false"],
      ["Sealed FEA", "false"],
      ["Standard-FEA", `${validationStatus}${validationTask.id == null ? "" : ` #${validationTask.id}`}`],
      ["Actual T120 gate", actualGates.pass === true ? "PASS" : actualGates.pass === false ? "FAIL" : "pending"],
    ].forEach(([name, value, className]) => {
      const box = element("div", className || "");
      box.append(element("span", "", name), element("b", "", value));
      summary.append(box);
    });

    appendDefinitionItems("#sealed-successor-provenance", [
      ["source_kind", payload.source_kind],
      ["lifecycle", `${payload.lifecycle_state} · terminal=${payload.terminal_result === true}`],
      ["evidence_id", payload.evidence_id],
      ["cohort_id", payload.cohort_id],
      ["constraint_version", payload.constraint_version],
      ["integrity_verified", String(payload.integrity_verified === true)],
      ["production / FEA", `production=${payload.production === true} · FEA=${payload.fea === true}/${payload.fea_status}`],
      ["scheduler task", payload.scheduler_task_id == null ? "none (FEA pending)" : payload.scheduler_task_id],
      ["handoff SHA-256", integrity.handoff_sha256],
      ["candidate SHA-256", integrity.candidate_sha256],
      ["bundle manifest SHA-256", integrity.bundle_manifest_sha256],
      ["deployment plan SHA-256", integrity.deployment_plan_sha256],
      ["replay SHA-256", integrity.replay_attestation_sha256],
      ["replay commit", integrity.replay_attestation_commit],
      ["ValidateOnly plan SHA-256", integrity.validateonly_plan_sha256],
      ["read-only UI SHA-256", integrity.read_only_ui_sha256],
      ["validation receipt SHA-256", integrity.validation_receipt_sha256],
      ["decoded FEA params", `${candidate.decoded_fea_param_count ?? "—"} · ${candidate.decoded_fea_params_sha256 || "—"}`],
      ["dedupe identity", candidate.dedupe_identity_sha256],
      ["source model SHA-256", model.source_model_manifest_sha256],
      ["deployment model SHA-256", model.deployment_model_manifest_sha256],
      ["NSGA code revision", model.nsga_code_revision],
      ["validation candidate digest", validation.candidate_digest],
      ["validation task", validationTask.id == null ? "not submitted" : `${validationTask.id} · ${validationTask.name}`],
      ["validation result SHA-256", resultIdentity.sha256],
    ]);
    appendDefinitionItems("#sealed-successor-standard-fea", [
      ["overlay integrity", validationVerified ? "PASS / read-only" : validation.configured === true ? "REJECTED / fail-closed" : "not configured"],
      ["runtime / submission", `${validation.runtime_state || "—"} / ${validation.submission_state || "—"}`],
      ["task id / name", validationTask.id == null ? "—" : `${validationTask.id} / ${validationTask.name || "—"}`],
      ["task status", validationTask.status || validation.task_status],
      ["collection state", validation.collection_state],
      ["collected / valid", `${validation.collected === true} / ${validation.valid === true}`],
      ["result available / contract", `${resultIdentity.available === true} / ${resultIdentity.contract_valid === true}`],
      ["result candidate identity", resultIdentity.candidate_identity_matches === true ? "MATCH" : resultIdentity.available === true ? "MISMATCH" : "pending"],
      ["result task / SHA-256", resultIdentity.available === true ? `${resultIdentity.task_id} / ${resultIdentity.sha256}` : "pending"],
      ["actual hard gates", actualGates.pass === true ? "PASS" : actualGates.pass === false ? `FAIL · ${(actualGates.reasons || []).join("; ") || "reason unavailable"}` : "pending"],
      ["actual Llt full", compact(measurements.Llt_full_uH, "µH")],
      ["actual max temperature", compact(measurements.maximum_temperature_C, "°C")],
      ["actual B design", compact(measurements.B_design_T, "T")],
      ["actual core / winding loss", `${compact(measurements.P_core_total_W, "W")} / ${compact(measurements.P_winding_total_W, "W")}`],
      ["full-model validation eligible", String(validation.full_model_validation_candidate_eligible === true)],
      ["overlay manifest SHA-256", validationIntegrity.manifest_sha256],
      ["overlay state / status SHA-256", `${validationIntegrity.state_sha256 || "—"} / ${validationIntegrity.status_sha256 || "—"}`],
      ["updated at", validation.updated_at],
      ["warning", validation.warning],
    ]);
    appendDefinitionItems(
      "#sealed-successor-standard-fea-temperatures",
      Object.entries(actual.temperatures || {}).map(([name, item]) => [
        name,
        item && item.applicable === false
          ? "not applicable"
          : `${compact(item && item.value_C, "°C")} · ${item && item.pass === true ? "PASS" : item && item.pass === false ? "FAIL" : "pending"}`,
      ]),
    );
    appendDefinitionItems("#sealed-successor-physics", [
      ["Llt μ", compact(llt.mu_uH, "µH")],
      ["Llt q90 half-width", compact(llt.q90_half_width_uH, "µH")],
      ["Llt robust G", `${compact(llt.G_uH, "µH")} · ${llt.pass === true ? "PASS" : "FAIL"}`],
      ["Llt target ± tolerance", `${compact(llt.target_uH, "µH")} ± ${compact(llt.tolerance_uH, "µH")}`],
      ["0.5×Lm resonance", compact(resonance.screen_Hz, "Hz")],
      ["resonance minimum", `${compact(resonance.minimum_Hz, "Hz")} · ${resonance.pass === true ? "PASS" : "FAIL"}`],
      ["objective volume", compact(candidate.objective_volume_L, "L")],
      ["objective total loss", compact(candidate.objective_total_loss_W, "W")],
      ["temperature limit", compact(candidate.temperature_limit_C, "°C")],
      ["hard size limit", `${hard.size_W_max_mm ?? "—"} × ${hard.size_L_max_mm ?? "—"} × ${hard.size_H_max_mm ?? "—"} mm`],
    ]);
    appendDefinitionItems(
      "#sealed-successor-temperatures",
      Object.entries(candidate.robust_temperatures_C || {}).map(([name, value]) => [
        name,
        `${compact(value, "°C")} · ${Number(value) <= Number(candidate.temperature_limit_C) ? "PASS" : "FAIL"}`,
      ]),
    );
    appendDefinitionItems(
      "#sealed-successor-constraints",
      Object.entries(candidate.constraints_G || {}).map(([name, value]) => [
        name,
        `${compact(value)} · ${Number(value) <= 0 ? "PASS" : "FAIL"}`,
      ]),
    );
    appendDefinitionItems(
      "#sealed-successor-geometry",
      Object.entries(candidate.decoded_geometry || {}).map(([name, value]) => {
        const unit = name.endsWith("_mm") ? "mm" : name.endsWith("_pct") ? "%" : "";
        return [name, compact(value, unit)];
      }),
    );
  }

  function showSealedSuccessor() {
    if (!state.sealedSuccessor) return;
    populateSealedSuccessorDialog(state.sealedSuccessor);
    $("#sealed-successor-dialog").showModal();
  }

  function nsgaScatterSeries(candidates = [], nearCandidates = []) {
    const isCoordinate = (value) => (
      value != null && value !== "" && Number.isFinite(Number(value))
    );
    const hasCoordinates = (item) => (
      isCoordinate(item?.volume_L)
      && isCoordinate(item?.total_loss_W)
    );
    const paretoPoints = (Array.isArray(candidates) ? candidates : [])
      .filter(hasCoordinates)
      .map((item) => ({ ...item, chart_kind: "pareto" }));
    const diagnosticPoints = (Array.isArray(nearCandidates) ? nearCandidates : [])
      .filter((item) => (
        hasCoordinates(item)
        && item.valid_pareto === false
        && item.production_eligible === false
        && item.fea_submission_approved === false
        && item.gui_launch_eligible === false
      ))
      .map((item) => ({ ...item, chart_kind: "near" }));
    return { paretoPoints, diagnosticPoints };
  }

  function renderScatter(candidates, nearCandidates = []) {
    const svg = $("#nsga-chart"); svg.replaceChildren();
    const { paretoPoints, diagnosticPoints } = nsgaScatterSeries(
      candidates,
      nearCandidates,
    );
    const points = [...paretoPoints, ...diagnosticPoints];
    if (!points.length) return;
    const box = chartBounds(svg); box.bottom = 38;
    let xMin = Math.min(...points.map((item) => Number(item.volume_L))), xMax = Math.max(...points.map((item) => Number(item.volume_L)));
    let yMin = Math.min(...points.map((item) => Number(item.total_loss_W))), yMax = Math.max(...points.map((item) => Number(item.total_loss_W)));
    if (xMin === xMax) { xMin -= 1; xMax += 1; } if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const xPad = (xMax - xMin) * .06, yPad = (yMax - yMin) * .08; xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
    const sx = (value) => box.left + (value - xMin) / (xMax - xMin) * (box.width - box.left - box.right);
    const sy = (value) => box.top + (1 - (value - yMin) / (yMax - yMin)) * (box.height - box.top - box.bottom);
    for (let i = 0; i <= 4; i += 1) {
      const ratio = i / 4, yValue = yMin + (yMax - yMin) * ratio, xValue = xMin + (xMax - xMin) * ratio;
      svg.append(svgElement("line", { x1: box.left, y1: sy(yValue), x2: box.width - box.right, y2: sy(yValue), class: "grid-line" }));
      const yt = svgElement("text", { x: box.left - 8, y: sy(yValue) + 3, "text-anchor": "end" }); yt.textContent = number(yValue, 0); svg.append(yt);
      const xt = svgElement("text", { x: sx(xValue), y: box.height - 10, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle" }); xt.textContent = number(xValue, 0); svg.append(xt);
    }
    const xLabel = svgElement("text", { x: box.width / 2, y: box.height - 1, "text-anchor": "middle" }); xLabel.textContent = "체적 [L]"; svg.append(xLabel);
    const yLabel = svgElement("text", { x: 12, y: box.height / 2, transform: `rotate(-90 12 ${box.height / 2})`, "text-anchor": "middle" }); yLabel.textContent = "총손실 [W]"; svg.append(yLabel);
    if (paretoPoints.length > 1) {
      const linePoints = [...paretoPoints]
        .sort((left, right) => Number(left.volume_L) - Number(right.volume_L))
        .map((item) => `${sx(Number(item.volume_L))},${sy(Number(item.total_loss_W))}`)
        .join(" ");
      svg.append(svgElement("polyline", {
        points: linePoints,
        class: "pareto-front-line",
      }));
    }
    points.forEach((candidate) => {
      const isNear = candidate.chart_kind === "near";
      const circle = svgElement("circle", {
        cx: sx(Number(candidate.volume_L)),
        cy: sy(Number(candidate.total_loss_W)),
        r: isNear || candidate.is_min_volume ? 5 : 3.2,
        class: `point${isNear ? " near" : candidate.is_min_volume ? " highlight" : ""}`,
        tabindex: 0,
      });
      const title = svgElement("title");
      title.textContent = `${isNear ? "근접 후보 · " : ""}${candidate.id} · ${number(candidate.volume_L, 1)} L · ${number(candidate.total_loss_W, 1)} W`;
      circle.append(title);
      circle.addEventListener("click", () => showCandidate(candidate));
      circle.addEventListener("keydown", (event) => { if (event.key === "Enter") showCandidate(candidate); });
      svg.append(circle);
    });
  }

  function constraintCheckLabel(key, check = {}) {
    const base = checkLabels[key] || key;
    const limit = check.limit;
    if (key === "resonance") {
      const minimum = Number(check.minimum_Hz);
      const maximum = Number(check.maximum_Hz);
      if (check.direction === "band"
          && Number.isFinite(minimum) && Number.isFinite(maximum)) {
        return `${base} ${number(minimum / 1000, 1)} kHz ≤ f < ${number(maximum / 1000, 1)} kHz`;
      }
      if (limit == null || limit === "" || !Number.isFinite(Number(limit))) {
        return base;
      }
      const operator = check.operator
        || (check.direction === "maximum" ? "<" : "≥");
      return `${base} ${operator}${number(Number(limit) / 1000, 1)} kHz`;
    }
    if (Array.isArray(limit) && limit.length === 2
        && limit.every((value) => value != null && value !== ""
          && Number.isFinite(Number(value)))) {
      return `${base} ${number(limit[0], 2)}–${number(limit[1], 2)} µH`;
    }
    if (limit == null || limit === "" || !Number.isFinite(Number(limit))) return base;
    const value = Number(limit);
    if (key === "temperature") return `${base} ≤${number(value, 1)}°C`;
    if (key === "bfield") return `${base} ≤${number(value, 2)} T`;
    if (key === "insulation") return `${base} ≥${number(value, 1)} mm`;
    if (key === "core_group") return `${base} ≤${number(value, 0)}`;
    if (key === "primary_thickness") return `${base} =${number(value, 1)} mm`;
    if (key.startsWith("size_")) return `${base} ≤${number(value, 0)} mm`;
    return base;
  }

  function makeChecks(container, checks) {
    container.replaceChildren();
    Object.entries(checks || {}).forEach(([key, check]) => {
      const status = check.pass === true ? "pass" : check.pass === false ? "fail" : "unknown";
      const item = element("div", `check-item ${status}`);
      item.append(element("span", "", constraintCheckLabel(key, check)));
      const value = check.value == null ? "확인 불가" : `${number(check.value, Math.abs(check.value) < 10 ? 3 : 1)} · ${labels[status]}`;
      item.append(element("b", "", value)); container.append(item);
    });
  }

  function makeNearViolationChecks(container, violationLabels) {
    container.replaceChildren();
    if (!violationLabels.length) {
      const item = element("div", "check-item unknown");
      item.append(
        element("span", "", "근접 후보 제약 상세"),
        element("b", "", "위반량 정보 없음"),
      );
      container.append(item);
      return;
    }
    violationLabels.forEach((label) => {
      const item = element("div", "check-item fail");
      item.append(
        element("span", "", "미충족 제약"),
        element("b", "", label),
      );
      container.append(item);
    });
  }

  function setLocalGuiButtons() {
    const candidate = state.selectedCandidate;
    const disabled = state.localGuiBusy || !candidate
      || candidate.artifact_hydrated !== true
      || candidate.gui_launch_eligible === false;
    const buildDisabled = disabled || candidate?.gui_build_eligible === false;
    const solveDisabled = disabled || candidate?.gui_solve_eligible === false;
    $("#open-symmetry-gui").disabled = buildDisabled;
    $("#solve-symmetry-gui").disabled = solveDisabled;
    $("#open-full-gui").disabled = buildDisabled;
    $("#solve-full-gui").disabled = solveDisabled;
    const reason = candidate?.gui_launch_limit_reason || "";
    $("#solve-symmetry-gui").title = solveDisabled ? reason : "";
    $("#solve-full-gui").title = solveDisabled ? reason : "";
  }

  function renderLocalGuiResult(launch) {
    const panel = $("#local-aedt-result");
    const validation = launch && launch.fea_validation && typeof launch.fea_validation === "object"
      ? launch.fea_validation : null;
    if (!launch || (!validation && !launch.error_summary && !launch.diagnosis)) {
      panel.classList.add("hidden");
      $("#local-aedt-result-summary").replaceChildren();
      $("#local-aedt-result-json").textContent = "";
      return;
    }
    panel.classList.remove("hidden");
    const summary = $("#local-aedt-result-summary");
    summary.replaceChildren();
    const result = validation && validation.result && typeof validation.result === "object"
      ? validation.result : {};
    [
      ["Validation", validation ? `${validation.status}${validation.success ? " · PASS" : " · FAIL"}` : (launch.state || "unknown")],
      ["Launch / candidate", `${launch.launch_id || "—"} / ${launch.candidate_id || "—"}`],
      ["Mode", `${launch.mode || "—"} · ${launch.action || "—"}`],
      ["Llt", result.Llt_phys == null ? "—" : compact(result.Llt_phys, "µH")],
      ["Core loss", result.P_core_total == null ? "—" : compact(result.P_core_total, "W")],
      ["Winding loss", result.P_winding_total == null ? "—" : compact(result.P_winding_total, "W")],
      ["B max", result.B_max_core == null ? "—" : compact(result.B_max_core, "T")],
      ["Project", validation?.project_name || "—"],
    ].forEach(([label, value]) => {
      const row = element("div");
      row.append(element("dt", "", label), element("dd", "", value));
      summary.append(row);
    });
    const reports = validation && Array.isArray(validation.inspection_reports)
      ? validation.inspection_reports : [];
    $("#local-aedt-report-summary").textContent = reports.length
      ? `AEDT Results reports: ${reports.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(", ")}`
      : "AEDT Results report evidence is not available yet.";
    $("#local-aedt-result-json").textContent = JSON.stringify({
      launch_id: launch.launch_id,
      candidate_id: launch.candidate_id,
      state: launch.state,
      launcher_active: launch.launcher_active,
      retained_aedt: launch.retained_aedt,
      occupied: launch.occupied,
      aedt_process_probe: launch.aedt_process_probe,
      exit_code: launch.exit_code,
      error_summary: launch.error_summary,
      diagnosis: launch.diagnosis,
      validation,
      result_path: launch.result_path,
      stdout_path: launch.stdout_path,
      stderr_path: launch.stderr_path,
    }, null, 2);
  }

  function renderLocalGuiStatus(launch, message = null, kind = "") {
    const status = $("#local-aedt-status");
    status.className = `local-aedt-status${kind ? ` ${kind}` : ""}`;
    const stateLabels = {
      completed_held: "FEA 완료 · AEDT/project 수동 종료 대기",
      failed_held: "FEA 실패 · AEDT/project 진단을 위해 유지",
      completed_detached: "FEA 완료 · AEDT 유지 확인 실패",
      ready_detached: "모델 생성 완료 · AEDT 유지 확인 실패",
      failed_detached: "FEA 실패 · AEDT 유지 확인 실패",
      failed_artifact: "결과 artifact 검증 실패",
      finalizing: "결과 기록 완료 · AEDT 유지 확인 중",
      finalizing_failed: "실패 기록 완료 · AEDT 유지 확인 중",
      building: "모델 생성 중", ready: "GUI 준비 완료 · AEDT에서 해석 시작 가능", exited: "프로세스 종료",
      solving: "EM·기생 C 해석 중 · 완료 전 GUI 조작 금지",
      failed: "실행 실패",
    };
    if (message) {
      status.textContent = message;
    } else if (launch) {
      const modelLabel = launch.mode === "full" ? "Full model" : "Symmetry model";
      const actionLabel = launch.action === "solve" ? "자동 해석" : "모델 생성";
      status.textContent = `${modelLabel} · ${actionLabel} · ${stateLabels[launch.state] || launch.state || "상태 확인 중"}`
        + `${launch.pid ? ` · PID ${launch.pid}` : ""}`
        + `${launch.started_at ? ` · ${dateTime(launch.started_at)}` : ""}`;
      if (launch.error_summary) status.textContent += ` · ${launch.error_summary}`;
      status.classList.add(["ready", "completed_held"].includes(launch.state) ? "success"
        : ["failed", "failed_held", "failed_detached", "failed_artifact",
          "completed_detached", "ready_detached"].includes(launch.state) ? "error" : "building");
    } else {
      status.textContent = "이 후보에 대해 로컬로 실행한 AEDT GUI가 없습니다.";
    }
    const details = $("#local-aedt-log-details");
    const output = launch
      ? [launch.stdout_tail, launch.stderr_tail].filter((value) => value && value.trim()).join("\n\n--- stderr ---\n")
      : "";
    $("#local-aedt-log").textContent = output;
    details.classList.toggle("hidden", !output);
    renderLocalGuiResult(launch);
  }

  async function refreshLocalGuiStatus() {
    const candidate = state.selectedCandidate;
    if (!candidate) return;
    try {
      const response = await fetch(
        `/api/local-aedt-gui/status?candidate_id=${encodeURIComponent(candidate.id)}`,
        { cache: "no-store", headers: { Accept: "application/json" } },
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (state.selectedCandidate !== candidate) return;
      renderLocalGuiStatus((payload.launches || [])[0] || null);
    } catch (error) {
      if (state.selectedCandidate === candidate) {
        renderLocalGuiStatus(null, `로컬 AEDT 상태 확인 실패: ${error.message}`, "error");
      }
    }
  }

  async function launchLocalGui(model, action = "build") {
    const candidate = state.selectedCandidate;
    if (!candidate || state.localGuiBusy) return;
    state.localGuiBusy = true;
    setLocalGuiButtons();
    renderLocalGuiStatus(
      null,
      `${model === "full" ? "Full" : "Symmetry"} model ${action === "solve" ? "해석" : "생성"} 요청 중…`,
      "building",
    );
    try {
      const response = await fetch("/api/operator/local-aedt-gui/launch", {
        method: "POST",
        cache: "no-store",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-MFT-Operator-Control": "local-aedt-gui-v1",
        },
        body: JSON.stringify({
          candidate_id: candidate.id,
          model,
          action,
          pareto_front_sha256: candidate.pareto_front_sha256 ?? null,
          pareto_row_number: candidate.pareto_row_number ?? null,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = payload.detail;
        if (detail && typeof detail === "object" && detail.launch) {
          renderLocalGuiStatus(detail.launch, detail.message || "이미 실행 중입니다.", "building");
          return;
        }
        throw new Error(typeof detail === "string" ? detail : `HTTP ${response.status}`);
      }
      renderLocalGuiStatus(payload.launch || null);
    } catch (error) {
      renderLocalGuiStatus(null, `로컬 AEDT 실행 실패: ${error.message}`, "error");
    } finally {
      state.localGuiBusy = false;
      setLocalGuiButtons();
    }
  }

  function showCandidate(candidate) {
    state.selectedCandidate = candidate;
    const dialogFacts = candidateDialogFacts(candidate);
    setLocalGuiButtons();
    if (state.localGuiPollTimer) {
      window.clearInterval(state.localGuiPollTimer);
      state.localGuiPollTimer = null;
    }
    if (dialogFacts.isReadOnlyNear) {
      renderLocalGuiStatus(
        null,
        "읽기 전용 근접 후보입니다. 자동 GUI/FEA 실행은 차단되며 아래 예측값과 미충족 제약을 확인할 수 있습니다.",
        "ready",
      );
    } else if (candidate.artifact_hydrated !== true) {
      renderLocalGuiStatus(null, "원본 Pareto 행이 아직 복원되지 않아 로컬 GUI 실행을 막았습니다.", "error");
    } else if (candidate.gui_launch_eligible === false) {
      renderLocalGuiStatus(
        null,
        "이전 조건 세대는 비교용 읽기 전용이며 로컬 GUI/FEA 조회와 실행을 차단했습니다.",
        "ready",
      );
    } else if (candidate.gui_solve_eligible === false) {
      renderLocalGuiStatus(
        null,
        candidate.gui_launch_limit_reason
          || "Symmetry/Full geometry build is available; local solve is blocked by the solver identity contract.",
        "ready",
      );
      refreshLocalGuiStatus();
      if (candidate.local_aedt_validation) {
        renderLocalGuiStatus(candidate.local_aedt_validation);
      }
    } else {
      renderLocalGuiStatus(null, "로컬 AEDT 실행 상태를 확인하는 중입니다.", "building");
      refreshLocalGuiStatus();
      if (candidate.local_aedt_validation) {
        renderLocalGuiStatus(candidate.local_aedt_validation);
      }
      state.localGuiPollTimer = window.setInterval(() => {
        if ($("#candidate-dialog").open) refreshLocalGuiStatus();
      }, 5000);
    }
    setText("#dialog-title", candidate.id);
    const sizeWidth = candidateNumber(candidate, "size_W_mm", "exterior_width_mm", "overall_width_mm");
    const sizeLength = candidateNumber(candidate, "size_L_mm", "exterior_length_mm", "overall_length_mm");
    const sizeHeight = candidateNumber(candidate, "size_H_mm", "exterior_height_mm", "overall_height_mm");
    const footprint = candidateNumber(candidate, "footprint_cm2", "exterior_footprint_cm2");
    const suppliedExteriorSize = candidateValue(candidate, "size_WxLxH_mm", "exterior_size_WxLxH_mm");
    const exteriorSize = suppliedExteriorSize != null
      ? `${suppliedExteriorSize}${/\bmm\b/i.test(String(suppliedExteriorSize)) ? "" : " mm"}`
      : [sizeWidth, sizeLength, sizeHeight].every((value) => value != null)
        ? `${compact(sizeWidth, "mm")} × ${compact(sizeLength, "mm")} × ${compact(sizeHeight, "mm")}`
        : "—";
    const temperatureRows = [
      ["Tx 권선 최고", candidateTemperature(candidate, "T_max_Tx")],
      ["Rx 중앙 권선 최고", candidateTemperature(candidate, "T_max_Rx_main")],
      ["Rx 측면 권선 최고", candidateTemperature(candidate, "T_max_Rx_side")],
      ["코어 전체 최고", candidateTemperature(candidate, "T_max_core")],
      ["Tx 권선 leeward 최고", candidateTemperature(candidate, "Tprobe_Tx_leeward_max")],
      ["Rx 중앙 권선 leeward 최고", candidateTemperature(candidate, "Tprobe_Rx_main_leeward_max")],
      ["Rx 측면 권선 leeward 최고", candidateTemperature(candidate, "Tprobe_Rx_side_leeward_max")],
      ["코어 중앙 영역 최고", candidateTemperature(candidate, "Tprobe_core_center_max")],
      ["코어 중앙 레그 최고", candidateTemperature(candidate, "Tprobe_core_center_leg_max")],
      ["코어 측면 레그 최고", candidateTemperature(candidate, "Tprobe_core_side_leg_max")],
      ["코어 상부 요크 최고", candidateTemperature(candidate, "Tprobe_core_top_yoke_max")],
    ];
    const suppliedMaxTemperature = dialogFacts.suppliedMaxTemperature;
    const knownTemperatures = temperatureRows.map(([, value]) => value).filter((value) => value != null);
    const maxTemperature = suppliedMaxTemperature ?? (knownTemperatures.length ? Math.max(...knownTemperatures) : null);
    const summary = $("#dialog-summary"); summary.replaceChildren();
    const primaryMainTurns = candidateNumber(candidate, "N1_main");
    const primarySideTurns = candidateNumber(candidate, "N1_side");
    const turnsPrimary = candidateNumber(candidate, "turns_primary")
      ?? (primaryMainTurns == null ? null : primaryMainTurns + (primarySideTurns ?? 0));
    const turnsSecondaryCenter = candidateNumber(candidate, "turns_secondary_center", "N2_main");
    const turnsSecondarySide = candidateNumber(candidate, "turns_secondary_side", "N2_side");
    const turns = [turnsPrimary, turnsSecondaryCenter, turnsSecondarySide]
      .every((value) => value != null)
      ? `${number(turnsPrimary)} / ${number(turnsSecondaryCenter)} / ${number(turnsSecondarySide)}`
      : "—";
    const predictedLeakage = dialogFacts.predictedLeakage;
    const leakageTarget = candidateNumber(candidate, "leakage_target_uH");
    const leakage = leakageTarget != null
      ? `${compact(predictedLeakage, "µH")} / 목표 ${compact(leakageTarget, "µH")}`
      : compact(predictedLeakage, "µH");
    [
      ["외형 폭 (W)", compact(sizeWidth, "mm"), "geometry-card"],
      ["외형 길이 (L)", compact(sizeLength, "mm"), "geometry-card"],
      ["외형 높이 (H)", compact(sizeHeight, "mm"), "geometry-card"],
      ["바닥면적 (W×L)", compact(footprint, "cm²"), "geometry-card"],
      ["체적", compact(candidateNumber(candidate, "volume_L"), "L")],
      ["총손실", compact(candidateNumber(candidate, "total_loss_W", "pred_total_loss_W"), "W")],
      ["누설 인덕턴스", leakage],
      ["최고 예측 온도", compact(maxTemperature, "°C")],
    ].forEach(([name, value, className]) => {
      const box = element("div", className || ""); box.append(element("span", "", name), element("b", "", value)); summary.append(box);
    });
    if (dialogFacts.isReadOnlyNear && !Object.keys(candidate.constraints || {}).length) {
      makeNearViolationChecks(
        $("#dialog-checks"),
        dialogFacts.violationLabels,
      );
    } else {
      makeChecks($("#dialog-checks"), candidate.constraints);
    }

    const appendDetails = (selector, items) => {
      const container = $(selector); container.replaceChildren();
      items.forEach(([label, value]) => {
        const item = element("div");
        item.append(element("dt", "", label), element("dd", "", value));
        container.append(item);
      });
    };
    appendDetails("#dialog-performance", [
      ["총손실", compact(candidateNumber(candidate, "total_loss_W", "pred_total_loss_W"), "W")],
      ["효율", compact(candidateNumber(candidate, "pred_efficiency_pct", "efficiency_pct"), "%")],
      ["정격 출력", compact(candidateNumber(candidate, "rated_power_W"), "W")],
      ["코어 손실", compact(candidateNumber(candidate, "pred_core_loss_W", "pred_P_core_total", "P_core_total"), "W")],
      ["전체 권선 손실", compact(candidateNumber(candidate, "pred_total_winding_loss_W", "pred_P_winding_total", "P_winding_total"), "W")],
      ["1차 권선 손실", compact(candidateNumber(candidate, "pred_primary_winding_loss_W", "pred_P_Tx_main_group", "P_Tx_main_group"), "W")],
      ["2차 중앙 권선 손실", compact(candidateNumber(candidate, "pred_secondary_center_winding_loss_W", "pred_P_Rx_main_group", "P_Rx_main_group"), "W")],
      ["2차 측면 권선 손실", compact(candidateNumber(candidate, "pred_secondary_side_winding_loss_W", "pred_P_Rx_side_total", "P_Rx_side_total"), "W")],
      ["2차 권선 손실 합계", compact(candidateNumber(candidate, "pred_secondary_winding_loss_W"), "W")],
      ["권선 성분 손실 합계", compact(candidateNumber(candidate, "pred_component_winding_loss_sum_W"), "W")],
      ["코어 콜드플레이트 손실", compact(candidateNumber(candidate, "pred_core_cold_plate_loss_W", "pred_P_core_plate_total", "P_core_plate_total"), "W")],
      ["권선 콜드플레이트 손실", compact(candidateNumber(candidate, "pred_winding_cold_plate_loss_W", "pred_P_wcp_total", "P_wcp_total"), "W")],
    ]);
    appendDetails("#dialog-temperatures", [
      ["전체 최고 예측", compact(maxTemperature, "°C")],
      ...temperatureRows.map(([label, value]) => [label, compact(value, "°C")]),
    ]);
    appendDetails("#dialog-electrostatic", [
      ["C_tx_tx", candidateCapacitance(candidate, "tx_tx")],
      ["C_rx_rx", candidateCapacitance(candidate, "rx_rx")],
      ["C_tx_rx", candidateCapacitance(candidate, "tx_rx")],
      ["Tx self 공진", candidateResonance(candidate, "tx_self")],
      ["Rx self 공진", candidateResonance(candidate, "rx_self")],
      ["상호권선 공진", candidateResonance(candidate, "interwinding")],
      ["0.5×Lm Tx screen", candidateResonance(candidate, "tx_screen")],
      ["0.5×Lm Rx screen", candidateResonance(candidate, "rx_screen")],
      ["0.5×Lm 최저 screen", candidateResonance(candidate, "min_screen")],
      ["0.5×Lm 최소 조건 / 여유", `≥ ${compact(candidateNumber(candidate, "resonance_minimum_required_Hz"), "Hz")} / +${compact(candidateNumber(candidate, "resonance_margin_Hz"), "Hz")}`],
    ]);
    appendDetails("#dialog-derived", [
      ["Validation badge", candidateValue(candidate, "validation_badge") || "??"],
      ["Hard-spec authority", candidateValue(candidate, "hard_spec_authority") || "??"],
      ["Validation tasks", candidateValue(candidate, "validation_task_summary") || "??"],
      ["Complete-Full pending", String(candidate.complete_full_pending === true)],
      ["Engineering approval", candidate.final_design_approved === true ? "APPROVED" : "NOT APPROVED"],
      ["Search source", candidate.tier1_source_label || candidate.result_lane],
      ["Search cohort", candidate.tier1_source_cohort_id],
      ["외형 크기 (W×L×H)", exteriorSize],
      ["턴수 (1차 / 2차 중앙 / 측면)", turns],
      ["설계 자속밀도", compact(candidateNumber(candidate, "B_design_analytic_T"), "T")],
      ["FEA 평균 B surrogate (참고)", compact(candidateNumber(candidate, "pred_B_mean_core"), "T")],
      ["1차 권선 1턴 두께", compact(candidateNumber(candidate, "cw1_conductor_thickness_mm", "cw1"), "mm")],
      ["2차 권선 1턴 두께", compact(candidateNumber(candidate, "cw2_conductor_thickness_mm", "cw2"), "mm")],
      ["1차 / 2차 턴 간격", `${compact(candidateNumber(candidate, "gap1_mm", "gap1"), "mm")} / ${compact(candidateNumber(candidate, "gap2_mm", "gap2"), "mm")}`],
      ["1차 중앙 권선팩 폭", compact(candidateNumber(candidate, "nwl1_main_pack_width_mm", "nwl1_main"), "mm")],
      ["1차 측면 권선팩 폭", compact(candidateNumber(candidate, "nwl1_side_pack_width_mm", "nwl1_side"), "mm")],
      ["2차 중앙 권선팩 폭", compact(candidateNumber(candidate, "nwl2_main_pack_width_mm", "nwl2_main"), "mm")],
      ["2차 측면 권선팩 폭", compact(candidateNumber(candidate, "nwl2_side_pack_width_mm", "nwl2_side"), "mm")],
      ["1차 / 2차 권선 높이", `${compact(candidateNumber(candidate, "nwh1_winding_height_mm", "nwh1"), "mm")} / ${compact(candidateNumber(candidate, "nwh2_winding_height_mm", "nwh2"), "mm")}`],
      ["코어 조 수 / 1조 깊이", `${number(candidateNumber(candidate, "n_core_group"))} / ${compact(candidateNumber(candidate, "core_depth_each_mm", "core_depth_each"), "mm")}`],
      ["코어 콜드플레이트 / 패드", `${compact(candidateNumber(candidate, "core_cold_plate_thickness_mm", "core_plate_t"), "mm")} / ${compact(candidateNumber(candidate, "core_thermal_pad_thickness_mm", "core_plate_pad_t"), "mm")}`],
      ["권선 콜드플레이트 / 패드", `${compact(candidateNumber(candidate, "winding_cold_plate_thickness_mm", "wcp_t"), "mm")} / ${compact(candidateNumber(candidate, "winding_thermal_pad_thickness_mm", "wcp_pad_t"), "mm")}`],
      ["TIM 열전도율", compact(candidateNumber(candidate, "thermal_pad_conductivity_W_mK"), "W/mK")],
      ["TIM 재료 정책", candidateValue(candidate, "thermal_pad_material_policy") || "—"],
      ["Cooling boundary", `${candidateValue(candidate, "fan_config") || "—"} fan · ${compact(candidateNumber(candidate, "fan_velocity_m_s"), "m/s")} · plate ${compact(candidateNumber(candidate, "plate_temp_C"), "°C")} · air ${compact(candidateNumber(candidate, "air_temp_C"), "°C")}`],
      ["Nominal size margins W / L / H", `${compact(candidateNumber(candidate, "nominal_width_margin_mm"), "mm")} / ${compact(candidateNumber(candidate, "nominal_length_margin_mm"), "mm")} / ${compact(candidateNumber(candidate, "nominal_height_margin_mm"), "mm")}`],
      ["Loss basis", candidateValue(candidate, "loss_basis_note") || "—"],
      ["Fan auxiliary power", candidateValue(candidate, "fan_auxiliary_power_note") || "—"],
      ["Tolerance qualification", candidateValue(candidate, "manufacturing_tolerance_note") || "—"],
      ["FEA 검증 수준", candidateValue(candidate, "fea_fidelity") || "—"],
      ["권선 콜드플레이트 길이", `${compact(candidateNumber(candidate, "wcp_len_pct"), "%")} / ${compact(candidateNumber(candidate, "wcp_len_x_mm", "wcp_len_x"), "mm")}`],
      ["코어 총 단면적", compact(candidateNumber(candidate, "Ae_gross_m2", "Ae_m2"), "m²")],
      ["코어 유효 단면적", compact(candidateNumber(candidate, "Ae_effective_m2"), "m²")],
      ["적층계수", compact(candidateNumber(candidate, "core_lamination_factor"))],
      ["초기 0.7식 B (감사용)", compact(candidateNumber(candidate, "B_legacy_0p7_T"), "T")],
      ["B 계산 기준", candidateValue(candidate, "B_area_basis") || "—"],
      ["B 파형", candidateValue(candidate, "B_design_waveform") || "—"],
      ["Surrogate 출력 기준", candidateValue(candidate, "surrogate_output_basis") || "—"],
    ]);
    const params = $("#dialog-parameters"); params.replaceChildren();
    const parameterUnits = {
      l1: "mm", l2: "mm", h1: "mm", w1: "mm", core_plate_t: "mm",
      core_plate_pad_t: "mm", cw1: "mm", gap1: "mm", cw2: "mm", gap2: "mm",
      nwh1: "mm", nwh2: "mm", wcp_t: "mm", wcp_pad_t: "mm",
      wcp_len_pct: "%", wcp_len_x: "mm",
      fan_velocity: "m/s", plate_temp: "°C", air_temp: "°C",
    };
    Object.entries(candidate.parameters || {}).forEach(([key, value]) => {
      const rendered = typeof value === "string"
        ? value
        : number(value, Number.isInteger(value) ? 0 : 3);
      const unit = parameterUnits[key] || "";
      const item = element("div"); item.append(element("dt", "", key), element("dd", "", `${rendered}${unit ? ` ${unit}` : ""}`)); params.append(item);
    });
    $("#candidate-dialog").showModal();
  }

  function renderVerification(payload) {
    setText("#verify-stage", payload.stage === "NOT_STARTED" ? "검증 전" : `${payload.stage} · round ${payload.round ?? "—"}`);
    const counts = payload.counts || {};
    setText(
      "#verify-coverage",
      counts.total
        ? `유효 ${number(counts.valid)} / ${number(counts.total)} · coverage ${counts.coverage == null ? "—" : `${number(counts.coverage * 100, 1)}%`}`
        : "기존 pipeline 검증 결과 없음 · 마감 캠페인 제출 상태는 위 별도 배너 기준",
    );
    const tbody = $("#verification-table"); tbody.replaceChildren();
    const candidates = [
      ...(payload.standard_candidates || []),
      ...(payload.fine_candidates || []),
    ];
    candidates.forEach((candidate) => {
      const evaluation = candidate.evaluation || {};
      const row = element("tr", "");
      row.append(element("td", "candidate-id", candidate.candidate_id));
      row.append(element("td", "table-value", candidate.task_id == null ? "—" : String(candidate.task_id)));
      row.append(element("td", "table-value", compact(evaluation.Llt_phys_uH, "µH")));
      row.append(element("td", "table-value", compact(evaluation.max_temperature_C, "°C")));
      row.append(element("td", "table-value", compact(evaluation.B_max_core_T, "T")));
      row.append(element("td", "table-value", compact(evaluation.total_loss_W, "W")));
      row.append(timingCell(evaluation.timing_seconds));
      const status = evaluation.computed_status || (candidate.outcome === "valid" ? "unknown" : "unknown");
      const cell = element("td"); cell.append(element("span", `state-chip ${status}`, labels[status])); row.append(cell); tbody.append(row);
    });
    $("#verification-empty").classList.toggle("hidden", candidates.length > 0);
    renderFinal(payload.final || {});
  }

  function renderFinal(final) {
    const status = final.status || "waiting";
    const statusClass = status === "pass" ? "pass" : ["fail", "blocked"].includes(status) ? "fail" : "waiting";
    const card = $("#final-card"); card.className = `panel final-card ${statusClass}`;
    setText("#final-title", status === "pass" ? "최종 설계 확정" : status === "fail" ? "최종 검증 실패" : status === "blocked" ? "최종 검증 차단" : "최종 검증 대기");
    const badge = $("#final-badge"); badge.className = `result-badge ${status === "pass" ? "pass" : ["fail", "blocked"].includes(status) ? "fail" : "unknown"}`; badge.textContent = status === "pass" ? "PASS" : status === "fail" ? "FAIL" : status === "blocked" ? "BLOCKED" : "WAIT";
    setText("#final-description", status === "blocked" ? (final.error || "작은 후보의 fine FEA 증거가 불완전해 최소부피 판정을 차단했습니다.") : final.available ? "명목 형상 full-model fine FEA의 항목별 판정입니다. 제작공차는 포함하지 않습니다." : "최소 체적 후보가 선정되고 fine FEA가 완료되면 항목별 판정이 고정됩니다.");
    makeChecks($("#final-checks"), final.evaluation?.checks || {});
    setText("#final-candidate", final.candidate_id);
    setText("#final-task", final.task_id);
    setText("#final-solver", final.evaluation?.solver_revision);
    setText("#final-library", final.evaluation?.library_revision);
    const timing = final.evaluation?.timing_seconds || {};
    setText("#final-time-matrix", duration(timing.matrix));
    setText("#final-time-loss", duration(timing.loss));
    setText("#final-time-icepak", duration(timing.icepak));
    setText("#final-time-total", duration(timing.total));
    setText("#final-error", final.error);
    $("#final-error").classList.toggle("hidden", !final.error);
  }

  function parallelStatus(message, kind = "") {
    const node = $("#parallel-target-status");
    node.textContent = message;
    node.className = `parallel-target-status${kind ? ` ${kind}` : ""}`;
  }

  function snapshotMessage(value) {
    if (value == null || value === "") return null;
    if (typeof value === "string") return value;
    if (typeof value === "object") return value.message || value.error || value.detail || null;
    return String(value);
  }

  function renderAedtAttach(scheduler = {}) {
    const attach = scheduler.aedt_attach && typeof scheduler.aedt_attach === "object"
      ? scheduler.aedt_attach
      : {};
    const license = attach.license && typeof attach.license === "object" ? attach.license : {};
    const pool = attach.pool && typeof attach.pool === "object" ? attach.pool : {};
    const nodeLocal = attach.node_local && typeof attach.node_local === "object" ? attach.node_local : {};
    const rawState = String(attach.state || "").toLowerCase();
    let stateKey = rawState;
    if (!stateKey) {
      if (pool.available === true && pool.enabled === true && pool.operational === true) stateKey = "healthy";
      else if (pool.available === true && pool.enabled === true) stateKey = "degraded";
      else if (pool.available === true && pool.enabled === false) stateKey = "disabled";
      else stateKey = attach.available === true ? "partial" : "unavailable";
    }
    const healthyStates = new Set(["healthy", "ready", "ok", "operational", "available"]);
    const warningStates = new Set(["degraded", "partial", "shortfall", "warming", "warning", "gated", "pool_unavailable"]);
    const errorStates = new Set(["error", "failed", "failure"]);
    const chipClass = healthyStates.has(stateKey) ? "pass"
      : warningStates.has(stateKey) ? "attention"
        : errorStates.has(stateKey) ? "fail" : "unknown";
    const stateLabels = {
      healthy: "정상", ready: "준비됨", ok: "정상", operational: "운영 중", available: "사용 가능",
      degraded: "주의", partial: "일부 확인", shortfall: "유휴 부족", warming: "예열 중", warning: "주의",
      gated: "Attach 제한", pool_unavailable: "Pool 확인 불가",
      disabled: "비활성", unavailable: "사용 불가", unknown: "확인 불가",
      error: "오류", failed: "오류", failure: "오류",
    };
    const stateChip = $("#aedt-attach-state");
    stateChip.className = `state-chip ${chipClass}`;
    stateChip.textContent = stateLabels[stateKey] || attach.state || "확인 불가";
    const card = $("#aedt-attach-card");
    card.className = `aedt-attach-card ${chipClass === "pass" ? "healthy" : chipClass === "attention" ? "degraded" : chipClass === "fail" ? "error" : "unavailable"}`;

    setText(
      "#aedt-license-usage",
      hasNumber(license.used) && hasNumber(license.total) ? `${number(license.used)} / ${number(license.total)}` : "—",
    );
    let poolState = "—";
    if (pool.available === true) {
      if (pool.enabled === false) poolState = "비활성";
      else if (stateKey === "warming") poolState = "예열 중";
      else if (stateKey === "shortfall") poolState = "유휴 부족";
      else if (pool.operational === true) poolState = "운영 중";
      else if (pool.enabled === true) poolState = "주의 필요";
      else poolState = "확인 불가";
    }
    setText("#aedt-pool-state", poolState);
    setText("#aedt-pool-idle", pool.available === true ? ratio(pool.idle_sessions, pool.min_idle_sessions) : "—");
    setText("#aedt-pool-sessions", pool.available === true ? ratio(pool.hard_sessions, pool.max_sessions) : "—");
    setText(
      "#aedt-pool-leases",
      pool.available === true && hasNumber(pool.live_leases) && hasNumber(pool.queued_leases)
        ? `${number(pool.live_leases)} / ${number(pool.queued_leases)}`
        : "—",
    );
    setText("#aedt-pool-capacity", pool.available === true ? ratio(pool.ready_sessions, pool.busy_sessions) : "—");

    const nodeLocalProgress = $("#aedt-node-local-progress");
    const activeHosts = Number(nodeLocal.active_host_tasks);
    const showNodeLocal = nodeLocal.available === true && Number.isFinite(activeHosts) && activeHosts > 0;
    nodeLocalProgress.classList.toggle("hidden", !showNodeLocal);
    if (showNodeLocal) {
      const bundleText = hasNumber(nodeLocal.bundle_count)
        ? `번들 ${number(nodeLocal.bundle_count)}개`
        : "번들 정보 없음";
      const projectText = hasNumber(nodeLocal.expected_projects)
        ? ` · 프로젝트 ${number(nodeLocal.expected_projects)}개`
        : "";
      const statuses = nodeLocal.statuses && typeof nodeLocal.statuses === "object" ? nodeLocal.statuses : {};
      const stateText = [
        ["Q", statuses.queued],
        ["A", statuses.attaching],
        ["R", statuses.running],
      ].filter(([, value]) => hasNumber(value)).map(([label, value]) => `${label} ${number(value)}`).join(" · ");
      nodeLocalProgress.textContent = `노드 로컬: 활성 호스트 ${number(activeHosts)}개 · ${bundleText}${projectText}${stateText ? ` · ${stateText}` : ""}`;
      const bundleIds = Array.isArray(nodeLocal.bundle_ids) ? nodeLocal.bundle_ids : [];
      nodeLocalProgress.title = bundleIds.length ? `번들: ${bundleIds.join(", ")}` : "";
    } else {
      nodeLocalProgress.textContent = "노드 로컬 AEDT 진행 정보 없음";
      nodeLocalProgress.title = "";
    }

    const errors = [
      ...(Array.isArray(attach.errors) ? attach.errors : []),
      license.error,
      pool.error,
    ].map(snapshotMessage).filter(Boolean);
    if (errors.length) {
      setText("#aedt-attach-detail", [...new Set(errors)].join(" · "));
    } else if (pool.warm_spare_reason) {
      setText("#aedt-attach-detail", pool.warm_spare_reason);
    } else if (attach.available === true) {
      setText("#aedt-attach-detail", license.checked_at ? `라이선스 ${dateTime(license.checked_at)} 확인` : "Scheduler AEDT snapshot 정상");
    } else {
      setText("#aedt-attach-detail", "Scheduler의 AEDT pool / license 정보를 사용할 수 없습니다.");
    }
  }

  function refillActionStatus(action) {
    const key = String(action || "").trim().toLowerCase();
    if (key === "no_refill_needed") return { label: "정상 (보충 불필요)", kind: "pass" };
    if (key === "pooled_bundle_pending") return { label: "AEDT 공유 번들 진행 중", kind: "attention" };
    if (key === "failed_closed") return { label: "오류로 안전정지 (관리자 확인 필요)", kind: "fail" };
    if (/(replac|submit|refill|reconcil|accept)/.test(key)) return { label: "보충 실행", kind: "checkpoint" };
    return { label: action || "확인 불가", kind: "unknown" };
  }

  function renderRefillController(refillController = {}) {
    const available = refillController.available === true;
    const actionStatus = refillActionStatus(refillController.action);
    const failedClosed = available && actionStatus.kind === "fail";
    const block = $("#refill-controller-status");
    block.classList.toggle("inline-error", failedClosed);
    const mode = $("#refill-controller-mode");
    mode.className = `state-chip ${failedClosed ? "fail" : available ? "pass" : "unknown"}`;
    setText(
      "#refill-controller-summary",
      available
        ? "자동 유지 모드 — 외부 컨트롤러가 MFT 동시 실행 수를 관리합니다."
        : "컨트롤러 상태 파일을 찾을 수 없음 — 스케줄러 수치는 상단 카드 참고",
    );
    $("#refill-controller-details").classList.toggle("hidden", !available);
    const action = $("#refill-controller-action");
    action.className = `state-chip ${actionStatus.kind}`;
    action.textContent = `${actionStatus.label} · ${relativeTickTime(refillController.last_tick_at ?? refillController.last_tick_time)}`;
    setText("#refill-controller-active", number(refillController.active_project_tasks_before));
    setText("#refill-controller-refilled", number(refillController.accepted_or_reconciled_count));
    const generationId = refillController.generation_id ?? refillController.generation?.id;
    setText("#refill-controller-generation", generationId == null ? null : String(generationId).slice(0, 12));
  }

  function renderParallelControl(scheduler = {}, refillController = state.dashboard?.refill_controller || {}) {
    const controlEnabled = scheduler.control_enabled === true;
    const policySupported = scheduler.policy_supported === true;
    const displayedTarget = scheduler.desired_simulations
      ?? scheduler.parallel_target
      ?? refillController.concurrency_target;
    setText("#parallel-current-target", number(displayedTarget));
    setText("#parallel-effective-target", number(scheduler.effective_simulations));
    setText("#parallel-validated-limit", number(scheduler.validated_concurrency_limit));
    setText("#parallel-logical-active", number(scheduler.logical_active));
    setText("#parallel-queued", number(scheduler.live_queued));
    setText("#parallel-attaching", number(scheduler.live_attaching));
    setText("#parallel-active", number(scheduler.live_active ?? scheduler.live_running));
    setText("#parallel-solving", number(scheduler.live_solving));
    renderAedtAttach(scheduler);
    $("#parallel-target-form").classList.remove("hidden");
    $("#refill-controller-status").classList.toggle("hidden", policySupported);
    $("#parallel-target-status").classList.remove("hidden");
    $("#parallel-control-note").classList.remove("hidden");
    setText("#parallel-control-eyebrow", policySupported ? "LEGACY PROJECT POLICY · NOT THE DEADLINE CAMPAIGN" : "LEGACY REFILL CONTROL · NOT THE DEADLINE CAMPAIGN");
    setText("#parallel-control-title", policySupported ? "기존 MFT 프로젝트 병렬 정책" : "기존 MFT 프로젝트 자동 실행 유지");
    setText(
      "#parallel-control-description",
      policySupported
        ? "이 0은 현재 512-seed Slurm 캠페인의 job 수가 아닙니다. 이 패널은 MFT_1MW_2026v1 simulation-policy만 표시하며, 현재 캠페인의 권위 있는 수치는 위 Codex 작업 카드에 표시됩니다."
        : "이 패널은 기존 refill-controller 범위입니다. 현재 512-seed Slurm 캠페인의 권위 있는 수치는 위 Codex 작업 카드에 표시됩니다.",
    );
    if (!policySupported) renderRefillController(refillController);
    const input = $("#parallel-target-input");
    const button = $("#parallel-target-button");
    const minimum = Number(scheduler.parallel_target_min);
    const maximum = Number(scheduler.parallel_target_max);
    const revision = scheduler.policy_revision;
    const enabled = scheduler.connected === true && controlEnabled
      && Number.isInteger(minimum) && Number.isInteger(maximum) && minimum <= maximum
      && (Number.isInteger(revision) || (typeof revision === "string" && revision.length > 0));
    input.min = Number.isInteger(minimum) ? String(minimum) : "0";
    if (Number.isInteger(maximum)) input.max = String(maximum);
    else input.removeAttribute("max");
    if (!state.parallelTargetDirty && document.activeElement !== input && displayedTarget != null) {
      input.value = String(displayedTarget);
    }
    input.disabled = !enabled || state.updatingParallelTarget;
    button.disabled = !enabled || state.updatingParallelTarget;
    const constraint = scheduler.resource_constraint;
    const constraintReason = constraint && typeof constraint === "object"
      ? (constraint.reason ?? constraint.detail ?? constraint.code)
      : constraint;
    if (state.updatingParallelTarget) {
      parallelStatus("새 목표를 Scheduler에 적용하는 중입니다.");
    } else if (scheduler.project_error) {
      parallelStatus(scheduler.project_error, "error");
    } else if (!enabled) {
      parallelStatus(
        scheduler.control_gate_reason
          || scheduler.error
          || "Scheduler simulation-policy 변경 gate가 열리지 않았습니다.",
        "error",
      );
    } else if (constraintReason) {
      parallelStatus(
        `Desired ${number(displayedTarget)} · effective ${number(scheduler.effective_simulations)} · 자원 제한: ${constraintReason}`,
      );
    } else {
      parallelStatus(
        `Desired ${number(displayedTarget)} · effective ${number(scheduler.effective_simulations)} · 검증 상한 ${number(scheduler.validated_concurrency_limit)} · loopback/신뢰 LAN 제어`,
      );
    }
  }

  function computeStageClass(stage = {}) {
    const value = String(stage.state || stage.task_status || "").toLowerCase();
    if (stage.complete === true || ["complete", "completed", "succeeded", "published", "active"].includes(value)) {
      return "pass";
    }
    if (stage.configured === true && (stage.available !== true
        || ["failed", "error", "rejected", "unavailable"].includes(value)
        || value.includes("failed")
        || value.includes("rejected"))) {
      return "fail";
    }
    return "unknown";
  }

  function renderComputeStage(key, stage = {}) {
    const chip = $(`#compute-${key}-state`);
    const detail = $(`#compute-${key}-detail`);
    const value = stage.state || stage.task_status
      || (stage.configured === false ? "준비 중" : "확인 중");
    chip.className = `state-chip ${computeStageClass(stage)}`;
    chip.textContent = value;
    if (stage.available !== true) {
      detail.textContent = stage.error || (stage.configured === false
        ? "후속 controller가 시작되면 자동으로 표시됩니다."
        : "상태 파일을 읽을 수 없습니다.");
      return;
    }
    if (key === "hpo") {
      detail.textContent = `results ${number(stage.authenticated_results || 0)} · trials ${number(stage.authenticated_trials || 0)}`
        + `${stage.task_id ? ` · task ${stage.task_id}` : ""}`
        + `${stage.generation_id ? ` · generation ${String(stage.generation_id).slice(0, 12)}` : ""}`;
      return;
    }
    detail.textContent = `${stage.task_id ? `task ${stage.task_id}` : "task 대기"}`
      + `${stage.task_status ? ` · ${stage.task_status}` : ""}`
      + `${stage.generation_id ? ` · generation ${String(stage.generation_id).slice(0, 12)}` : ""}`
      + `${stage.failure_message ? ` · ${stage.failure_message}` : ""}`
      + `${stage.advisory ? ` · advisory: ${stage.advisory}` : ""}`;
  }

  function renderComputeCampaign(payload = {}) {
    const stages = payload.stages || {};
    ["hpo", "acceptance", "harvest", "cutover"].forEach((key) => {
      renderComputeStage(key, stages[key] || {});
    });
    const configured = Object.values(stages).filter((stage) => stage?.configured === true);
    const failed = configured.some((stage) => computeStageClass(stage) === "fail");
    const allComplete = configured.length === 4
      && configured.every((stage) => computeStageClass(stage) === "pass");
    const stateChip = $("#compute-campaign-state");
    stateChip.className = `status-pill ${failed ? "status-error" : allComplete ? "status-active" : "status-warning"}`;
    stateChip.innerHTML = `<i></i> ${failed ? "확인 필요" : allComplete ? "전환 완료" : "병렬 진행 중"}`;
    setText(
      "#compute-campaign-detail",
      failed
        ? "구성된 단계 중 읽기 또는 실행 오류가 있습니다."
        : "탐색 자원을 유지하면서 학습 자원을 자연 완료 슬롯과 교환합니다.",
    );
    const resources = payload.resources || {};
    const policyTarget = hasNumber(resources.policy_active_target)
      ? resources.policy_active_target
      : resources.active_target;
    setText("#compute-resource-tasks", hasNumber(policyTarget) ? `target ${number(policyTarget)}` : "—");
    setText("#compute-resource-cpus", hasNumber(resources.requested_cpus) ? number(resources.requested_cpus) : "—");
    setText(
      "#compute-resource-memory",
      hasNumber(resources.requested_memory_mb)
        ? `${number(Number(resources.requested_memory_mb) / 1048576, 2)} TiB`
        : "—",
    );
    setText(
      "#compute-resource-per-task",
      hasNumber(resources.cpus_per_task) && hasNumber(resources.memory_mb_per_task)
        ? `${number(resources.cpus_per_task)} CPU / ${number(Number(resources.memory_mb_per_task) / 1024)} GiB`
        : "—",
    );
    setText(
      "#compute-resource-unplaced",
      hasNumber(resources.global_unplaced_limit)
        ? number(resources.global_unplaced_limit)
        : "—",
    );
  }

  function renderDiagnostics(payload) {
    const scheduler = payload.scheduler || {};
    const details = $("#scheduler-details"); details.replaceChildren();
    [["연결", scheduler.connected ? "정상" : "실패"], ["실행 / 대기", `${number(scheduler.running)} / ${number(scheduler.pending)}`], ["완료 / 실패", `${number(scheduler.completed)} / ${number(scheduler.failed)}`], ["조회 범위", scheduler.project || scheduler.task_prefix || "—"], ["모드", scheduler.control_enabled ? "versioned LAN policy control" : "GET only"]].forEach(([key, value]) => {
      const item = element("div"); item.append(element("dt", "", key), element("dd", "", value)); details.append(item);
    });
    const warnings = [
      ...(payload.data?.warnings || []), ...(payload.models?.warnings || []),
      ...(payload.nsga2?.warnings || []), ...(payload.verification?.warnings || []),
      ...(payload.continuous_pipeline?.warnings || []),
      ...(payload.nsga2_progress?.error ? [payload.nsga2_progress.error] : []),
      ...(payload.local_aedt_gui_launches?.error
        ? [payload.local_aedt_gui_launches.error] : []),
      ...(payload.compute_campaign?.warnings || []),
      ...(scheduler.error ? [scheduler.error] : []), ...(scheduler.project_error ? [scheduler.project_error] : []),
    ];
    const list = $("#artifact-warnings"); list.replaceChildren();
    [...new Set(warnings)].forEach((warning) => list.append(element("li", "", warning)));
    if (!warnings.length) list.append(element("li", "", "없음"));
  }

  function render(payload) {
    state.dashboard = payload;
    renderOverall(payload);
    renderContinuousPipeline(payload.continuous_pipeline || {});
    renderComputeCampaign(payload.compute_campaign || {});
    renderParallelControl(payload.scheduler || {}, payload.refill_controller || {});
    renderData(payload.data || {});
    renderModels(payload.models || { models: [] });
    renderNsga(payload.nsga2 || { candidates: [] }, payload.nsga2_progress);
    renderVerification(payload.verification || { counts: {}, final: {} });
    renderDiagnostics(payload);
  }

  function codexTimeRemaining(deadlineAt) {
    const deadline = new Date(deadlineAt);
    if (Number.isNaN(deadline.getTime())) return "마감 —";
    const remainingMs = deadline.getTime() - Date.now();
    const absoluteMinutes = Math.floor(Math.abs(remainingMs) / 60000);
    const days = Math.floor(absoluteMinutes / 1440);
    const hours = Math.floor((absoluteMinutes % 1440) / 60);
    const minutes = absoluteMinutes % 60;
    const compact = days > 0
      ? `${days}일 ${hours}시간`
      : `${hours}시간 ${minutes}분`;
    return remainingMs >= 0 ? `마감까지 ${compact}` : `마감 ${compact} 경과`;
  }

  function deadlineCampaignVerified(campaign = {}) {
    const allocationJobs = campaign.allocation_jobs_active;
    const submitted = campaign.submitted_total;
    const running = campaign.running;
    const collections = campaign.collections;
    return campaign.available === true
      && campaign.integrity_verified === true
      && Number.isInteger(allocationJobs)
      && Number.isInteger(submitted)
      && Number.isInteger(running)
      && Number.isInteger(collections)
      && allocationJobs >= 0
      && submitted >= 0
      && running >= 0
      && collections >= 0
      && running <= submitted
      && collections <= submitted
      && (running === 0 || allocationJobs > 0)
      && campaign.collection_zero_means_submission_zero === false;
  }

  function renderDeadlineCampaign(campaign = {}) {
    const verified = deadlineCampaignVerified(campaign);
    const allocationJobs = verified
      ? Number(campaign.allocation_jobs_active)
      : null;
    const submitted = verified ? Number(campaign.submitted_total) : null;
    const running = verified ? Number(campaign.running) : null;
    const collections = verified ? Number(campaign.collections) : null;
    const stateClass = verified
      ? (running > 0 ? "running" : "verified")
      : "unavailable";
    const topStrip = $("#codex-campaign-strip");
    const verificationStrip = $("#verify-deadline-campaign");
    topStrip.className = `codex-campaign-strip ${stateClass}`;
    verificationStrip.className = `verify-deadline-campaign ${stateClass}`;
    setText(
      "#codex-campaign-allocation-jobs",
      verified ? number(allocationJobs) : "—",
    );
    setText("#codex-campaign-submitted", verified ? number(submitted) : "—");
    setText("#codex-campaign-running", verified ? number(running) : "—");
    setText("#codex-campaign-collections", verified ? number(collections) : "—");
    if (!verified) {
      setText(
        "#codex-campaign-note",
        "마감 캠페인 권위 상태를 확인할 수 없습니다. legacy 정책의 0을 job 수로 해석하지 마십시오.",
      );
      setText("#verify-campaign-summary", "마감 캠페인 권위 상태 확인 불가");
      setText(
        "#verify-campaign-note",
        "아래 기존 pipeline 검증 표와 별도 범위이며, legacy 정책 0은 제출 건수가 아닙니다.",
      );
      return;
    }
    const summary = `Slurm allocation job ${number(allocationJobs)} · 제출 ${number(submitted)} · 실행 ${number(running)} · 완료·회수 ${number(collections)}`;
    setText("#verify-campaign-summary", summary);
    if (collections === 0 && submitted > 0) {
      setText(
        "#codex-campaign-note",
        `권위 있는 마감 캠페인 상태 · Slurm allocation job ${number(allocationJobs)}개에서 task ${number(running)}건 실행 중 · 완료·회수 0은 제출 0이 아닙니다.`,
      );
      setText(
        "#verify-campaign-note",
        "완료·회수 0은 아직 회수된 FEA 결과가 0건이라는 뜻이며, job 제출 0건이라는 뜻이 아닙니다. 아래 표는 기존 pipeline 범위입니다.",
      );
      return;
    }
    setText(
      "#codex-campaign-note",
      "Codex가 발행한 읽기 전용 마감 캠페인 상태이며, 아래 legacy 정책 카운터와 별도 범위입니다.",
    );
    setText(
      "#verify-campaign-note",
      "마감 캠페인의 제출·실행·회수 수치입니다. 아래 표는 기존 pipeline 범위입니다.",
    );
  }

  function renderCodexWorkItem(item = {}) {
    const card = element("article", `codex-work-item ${item.state || ""}`);
    const header = element("div", "codex-work-item-heading");
    header.append(
      element("strong", "", item.title || item.id || "제목 없음"),
      element("time", "", dateTime(item.updated_at)),
    );
    card.append(header, element("p", "", item.detail || "상세 설명 없음"));
    if (
      item.progress_pct != null
      && Number.isFinite(Number(item.progress_pct))
    ) {
      const progress = element("div", "codex-work-progress");
      const bar = element("span");
      bar.style.width = `${Math.max(
        0,
        Math.min(100, Number(item.progress_pct)),
      )}%`;
      progress.append(bar);
      progress.setAttribute(
        "aria-label",
        `진척도 ${Number(item.progress_pct)}%`,
      );
      card.append(progress);
    }
    const evidence = Array.isArray(item.evidence) ? item.evidence : [];
    if (evidence.length) {
      const details = element("details", "codex-work-evidence");
      details.append(element("summary", "", `근거 ${evidence.length}개`));
      const list = element("ul");
      evidence.forEach((value) => (
        list.append(element("li", "", String(value)))
      ));
      details.append(list);
      card.append(details);
    }
    return card;
  }

  function renderCodexWorkGroup(group, items) {
    const values = Array.isArray(items) ? items : [];
    setText(`#codex-${group}-count`, number(values.length));
    const list = $(`#codex-${group}-list`);
    list.replaceChildren();
    if (!values.length) {
      list.append(element("p", "codex-work-empty", "항목 없음"));
      return;
    }
    values.forEach((item) => list.append(renderCodexWorkItem(item)));
  }

  function renderCodexWork(payload = {}) {
    state.codexWork = payload;
    const panel = $("#codex-work-panel");
    const freshness = $("#codex-work-freshness");
    const verified = payload.available === true
      && payload.integrity_verified === true;
    panel.classList.toggle("loading", !verified);
    panel.classList.toggle("unavailable", !verified);
    panel.classList.toggle("stale", verified && payload.stale === true);
    if (!verified) {
      setText(
        "#codex-work-summary",
        payload.error || "Codex 상태 아티팩트를 사용할 수 없습니다.",
      );
      freshness.className = "state-chip fail";
      freshness.textContent = "상태 확인 불가";
      setText("#codex-work-deadline", "마감 —");
      setText("#codex-work-updated", "갱신 —");
      renderDeadlineCampaign({});
      renderCodexWorkGroup("current", []);
      renderCodexWorkGroup("completed", []);
      renderCodexWorkGroup("attention", []);
      return;
    }
    setText("#codex-work-title", payload.goal || "Codex 작업 현황");
    setText("#codex-work-summary", payload.summary || "요약 없음");
    setText("#codex-work-deadline", codexTimeRemaining(payload.deadline_at));
    setText("#codex-work-updated", `${dateTime(payload.generated_at)} 갱신`);
    freshness.className = `state-chip ${payload.stale ? "attention" : "pass"}`;
    freshness.textContent = payload.stale
      ? `오래된 상태 · ${duration(payload.age_seconds)}`
      : `최신 · ${duration(payload.age_seconds)} 전`;
    renderDeadlineCampaign(payload.deadline_campaign || {});
    renderCodexWorkGroup("current", payload.current);
    renderCodexWorkGroup("completed", payload.completed);
    renderCodexWorkGroup("attention", payload.attention);
  }

  function nsgaSourceIdentity(source = {}) {
    return [
      source.role || "",
      source.label || "",
      source.cohort_id || "",
      source.constraint_version || "",
      source.hard_spec_sha256 || "",
      source.model_manifest_sha256 || "",
      source.deployment_model_manifest_sha256 || "",
      source.nsga_code_revision || "",
      source.accepted === true ? "accepted" : "rejected",
    ].join("|");
  }

  function nsgaSearchAuthority(rootPayload = {}) {
    const authority = rootPayload?.search_authority || {};
    if (authority.kind === "current7") {
      return {
        kind: "current7",
        authority,
        search: rootPayload?.tier1_current7_search || {},
      };
    }
    return {
      kind: "legacy",
      authority,
      search: rootPayload?.tier1_feedback_search || {},
    };
  }

  function nsgaAuthoritySearch(rootPayload = {}) {
    return nsgaSearchAuthority(rootPayload).search;
  }

  function current7CandidateMatches(candidate, search) {
    return candidate?.tier1_source_role === "current7"
      && candidate?.model_lane === "corrected_current7"
      && candidate?.model_id === `tier1-current7:${search.bundle_id}`
      && candidate?.tier1_source_cohort_id === search.cohort_id
      && candidate?.embedded_authenticated_result === true
      && candidate?.production_eligible === false
      && candidate?.fea_submission_approved === false
      && candidate?.gui_launch_eligible === false;
  }

  function current7NearCandidateMatches(candidate, search) {
    return candidate?.source_role === "current7"
      && candidate?.source_cohort_id === search.cohort_id;
  }

  function nsgaAuthorityView(rootPayload = {}) {
    const selected = nsgaSearchAuthority(rootPayload);
    if (selected.kind !== "current7") return rootPayload;
    const { authority, search } = selected;
    const ready = authority.configured === true
      && authority.available === true
      && authority.integrity_verified === true
      && search.available === true
      && search.integrity_verified === true;
    const candidates = ready && Array.isArray(rootPayload.candidates)
      ? rootPayload.candidates.filter((candidate) => (
        current7CandidateMatches(candidate, search)
      ))
      : [];
    const nearCandidates = ready && Array.isArray(rootPayload.near_feasible_preview)
      ? rootPayload.near_feasible_preview.filter((candidate) => (
        current7NearCandidateMatches(candidate, search)
      ))
      : [];
    const validCandidates = candidates.filter((candidate) => (
      candidate?.spec_status === "pass"
    ));
    const volumes = validCandidates
      .map((candidate) => Number(candidate.volume_L))
      .filter(Number.isFinite);
    const losses = validCandidates
      .map((candidate) => Number(candidate.total_loss_W))
      .filter(Number.isFinite);
    const completed = Number(search.completed_count);
    const running = Number(search.running_count);
    return {
      ...rootPayload,
      available: ready,
      status: ready && running > 0
        ? "running"
        : ready && (completed > 0 || candidates.length > 0)
          ? "completed"
          : "waiting",
      result_scope: candidates.length > 0 ? "current_model" : "waiting_current",
      candidate_count: candidates.length,
      display_candidate_count: candidates.length,
      valid_candidate_count: validCandidates.length,
      candidates,
      near_feasible_preview: nearCandidates,
      near_feasible_preview_count: nearCandidates.length,
      selected_model_id: ready ? `tier1-current7:${search.bundle_id}` : null,
      constraint_version: search.constraint_version || null,
      constraints: search.constraints || {},
      summary: {
        ...(rootPayload.summary || {}),
        candidate_count: candidates.length,
        display_candidate_count: candidates.length,
        valid_candidate_count: validCandidates.length,
        min_volume_L: volumes.length > 0 ? Math.min(...volumes) : null,
        min_loss_W: losses.length > 0 ? Math.min(...losses) : null,
      },
      note: ready
        ? rootPayload.note
        : authority.warning
          || "Current7 search authority is unavailable; legacy Pareto remains archive-only.",
    };
  }

  function exactNsgaIdentity(left, right) {
    return typeof left === "string" && left.length > 0 && left === right;
  }

  function exactNsgaCount(search, searchKey, progress, progressKey = searchKey) {
    return Number.isInteger(search[searchKey]) && search[searchKey] >= 0
      && Number.isInteger(progress[progressKey]) && progress[progressKey] >= 0
      && search[searchKey] === progress[progressKey];
  }

  function current7ProgressMatches(search = {}, progress = {}) {
    if (progress.authority_kind !== "current7"
        || progress.coherent_snapshot !== true) return false;
    const identityPairs = [
      [search.bundle_manifest_sha256, progress.authority_identity_sha256],
      [search.index_file_sha256, progress.authority_index_file_sha256],
      [search.snapshot_file_sha256, progress.authority_snapshot_sha256],
      [search.snapshot_sha256, progress.authority_snapshot_identity_sha256],
      [search.constraint_identity_sha256, progress.constraint_identity_sha256],
      [search.constraint_version, progress.constraint_version],
    ];
    if (!identityPairs.every(([left, right]) => exactNsgaIdentity(left, right))) {
      return false;
    }
    const exactCountPairs = [
      ["search_count", "search_count"],
      ["running_count", "running_count"],
      ["queued_count", "queued_count"],
      ["attaching_count", "attaching_count"],
      ["active_plus_queued", "active_plus_queued"],
      ["completed_count", "completed_count"],
      ["failed_count", "failed_count"],
      ["cancelled_count", "cancelled_count"],
      ["timeout_count", "timeout_count"],
      ["authenticated_terminal_seed_count", "authenticated_terminal_seed_count"],
      ["authenticated_terminal_seed_count", "terminal_results_verified"],
      ["refused_terminal_count", "refused_terminal_count"],
      ["refused_terminal_count", "failed_terminal_results_verified"],
      ["feasible_pareto_count", "feasible_pareto_count"],
    ];
    if (!exactCountPairs.every(([searchKey, progressKey]) => (
      exactNsgaCount(search, searchKey, progress, progressKey)
    ))) return false;
    return Number(progress.active_plus_queued)
      === Number(progress.running_count)
        + Number(progress.queued_count)
        + Number(progress.attaching_count);
  }

  function nsgaProgressMatchesDashboard(rootPayload = {}, progress = {}) {
    const authority = nsgaSearchAuthority(rootPayload);
    const search = authority.search;
    if (search.available !== true || search.integrity_verified !== true
        || progress.available !== true || progress.integrity_verified !== true
        || progress.counters_consistent !== true) return false;
    if (authority.kind === "current7") {
      if (authority.authority.configured !== true
          || authority.authority.available !== true
          || authority.authority.integrity_verified !== true) return false;
      return current7ProgressMatches(search, progress);
    }
    const identityKeys = [
      "constraint_version",
      "model_manifest_sha256",
      "deployment_model_manifest_sha256",
    ];
    if (!identityKeys.every((key) => (
      typeof search[key] === "string" && search[key].length > 0
      && search[key] === progress[key]
    ))) return false;
    const countKeys = [
      "source_count",
      "supplemental_source_count",
      "configured_source_count",
      "rejected_source_count",
    ];
    if (!countKeys.every((key) => (
      Number.isInteger(Number(search[key]))
      && Number(search[key]) === Number(progress[key])
    ))) return false;
    const searchSources = Array.isArray(search.sources) ? search.sources : [];
    const progressSources = Array.isArray(progress.sources) ? progress.sources : [];
    return searchSources.length === Number(search.source_count)
      && progressSources.length === Number(progress.source_count)
      && searchSources.map(nsgaSourceIdentity).join("\n")
        === progressSources.map(nsgaSourceIdentity).join("\n");
  }

  function nsgaProgressProjection(rootPayload, progress) {
    if (!progress) return null;
    if (progress.integrity_verified === true) {
      return nsgaProgressMatchesDashboard(rootPayload, progress)
        ? progress
        : {
          available: false,
          integrity_verified: false,
          error: "NSGA progress snapshot identity does not match the current dashboard; waiting for a coherent refresh",
        };
    }
    return progress;
  }

  async function fetchJsonWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    let timedOut = false;
    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      if (timedOut) {
        throw new Error(`request timed out after ${Math.round(timeoutMs / 1000)}s`);
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function renderAuxiliarySnapshots() {
    if (!state.dashboard) return;
    const progress = nsgaProgressProjection(
      state.dashboard.nsga2 || {},
      state.nsgaProgress,
    );
    state.dashboard.nsga2_progress = progress;
    state.dashboard.local_aedt_gui_launches = state.localGuiLaunches;
    renderNsga(state.dashboard.nsga2 || { candidates: [] }, progress);
    renderDiagnostics(state.dashboard);
  }

  function refreshNsgaProgress(options) {
    if (state.nsgaProgressRequest) return state.nsgaProgressRequest;
    if (!state.nsgaProgress) {
      state.nsgaProgress = { available: false, loading: true, error: null };
    }
    state.nsgaProgressRequest = (async () => {
      try {
        state.nsgaProgress = await fetchJsonWithTimeout(
          "/api/nsga2/progress",
          options,
          nsgaProgressRequestTimeoutMs,
        );
      } catch (error) {
        state.nsgaProgress = {
          available: false,
          integrity_verified: false,
          error: `NSGA progress endpoint unavailable: ${error.message}`,
        };
      }
      renderAuxiliarySnapshots();
    })().finally(() => {
      state.nsgaProgressRequest = null;
    });
    return state.nsgaProgressRequest;
  }

  function refreshLocalGuiLaunches(options) {
    if (state.localGuiLaunchesRequest) return state.localGuiLaunchesRequest;
    state.localGuiLaunchesRequest = (async () => {
      try {
        state.localGuiLaunches = await fetchJsonWithTimeout(
          "/api/local-aedt-gui/launches",
          options,
          localGuiRequestTimeoutMs,
        );
      } catch (error) {
        state.localGuiLaunches = {
          available: false,
          error: `local GUI launches endpoint unavailable: ${error.message}`,
        };
      }
      renderAuxiliarySnapshots();
    })().finally(() => {
      state.localGuiLaunchesRequest = null;
    });
    return state.localGuiLaunchesRequest;
  }

  function refreshCodexWork(options) {
    if (state.codexWorkRequest) return state.codexWorkRequest;
    state.codexWorkRequest = (async () => {
      try {
        const payload = await fetchJsonWithTimeout(
          "/api/codex-work",
          options,
          codexWorkRequestTimeoutMs,
        );
        renderCodexWork(payload);
      } catch (error) {
        renderCodexWork({
          available: false,
          integrity_verified: false,
          error: `Codex 작업 상태 endpoint unavailable: ${error.message}`,
        });
      }
    })().finally(() => {
      state.codexWorkRequest = null;
    });
    return state.codexWorkRequest;
  }

  async function refresh() {
    if (state.refreshing) return;
    state.refreshing = true;
    $("#refresh-button").disabled = true;
    $("#loading-indicator").classList.add("loading");
    try {
      const options = { cache: "no-store", headers: { Accept: "application/json" } };
      refreshCodexWork(options);
      refreshNsgaProgress(options);
      refreshLocalGuiLaunches(options);
      const payload = await fetchJsonWithTimeout(
        "/api/dashboard",
        options,
        dashboardRequestTimeoutMs,
      );
      if (payload.error || !payload.data) throw new Error(payload.error || "dashboard payload is incomplete");
      payload.nsga2_progress = nsgaProgressProjection(
        payload.nsga2 || {},
        state.nsgaProgress,
      );
      payload.local_aedt_gui_launches = state.localGuiLaunches;
      render(payload);
    } catch (error) {
      const fallback = {
        generated_at: new Date().toISOString(),
        status: { overall: "error", current_stage_label: "모니터 연결 실패", warnings: [`WEB UI 데이터를 불러오지 못했습니다: ${error.message}`], stages: [] },
      };
      renderOverall(fallback);
    } finally {
      state.refreshing = false;
      $("#refresh-button").disabled = false;
      $("#loading-indicator").classList.remove("loading");
    }
  }

  if (!hasDOM) {
    if (typeof module !== "undefined" && module.exports) {
      module.exports = {
        candidateCapacitance,
        candidateDialogFacts,
        candidateNumber,
        candidateResonance,
        candidateTemperature,
        candidateValue,
        deadlineCampaignVerified,
        nearViolationLabel,
        nsgaScatterSeries,
        beginNsgaGenerationRequest,
        nsgaGenerationCacheKey,
        nsgaGenerationDetailVerified,
        settleNsgaGenerationRequest,
        nsgaAuthoritySearch,
        nsgaAuthorityView,
        nsgaProgressMatchesDashboard,
      };
    }
    return;
  }

  $("#refresh-button").addEventListener("click", refresh);
  $("#sealed-successor-detail-button").addEventListener("click", showSealedSuccessor);
  $("#open-symmetry-gui").addEventListener("click", () => launchLocalGui("symmetry", "build"));
  $("#open-full-gui").addEventListener("click", () => launchLocalGui("full", "build"));
  $("#solve-symmetry-gui").addEventListener("click", () => {
    if (window.confirm("Symmetry 모델의 Maxwell EM·기생 C 해석을 지금 로컬에서 시작할까요? 완료될 때까지 AEDT GUI를 조작하지 마세요.")) {
      launchLocalGui("symmetry", "solve");
    }
  });
  $("#solve-full-gui").addEventListener("click", () => {
    if (window.confirm("Full 모델 해석은 오래 걸리고 로컬 CPU·AEDT 라이선스를 사용합니다. Maxwell EM·기생 C 해석을 시작할까요?")) {
      launchLocalGui("full", "solve");
    }
  });
  $("#candidate-dialog").addEventListener("close", () => {
    state.selectedCandidate = null;
    if (state.localGuiPollTimer) window.clearInterval(state.localGuiPollTimer);
    state.localGuiPollTimer = null;
  });
  $("#model-history-metric").addEventListener("change", (event) => {
    const metric = event.target.value;
    if (!historyMetrics[metric]) return;
    state.historyMetric = metric;
    if (state.selectedModelData) renderModelHistory(state.selectedModelData);
  });
  $("#parallel-target-input").addEventListener("input", () => {
    state.parallelTargetDirty = true;
  });
  $("#parallel-target-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.updatingParallelTarget) return;
    const input = $("#parallel-target-input");
    const target = Number(input.value);
    const min = Number(input.min);
    const max = Number(input.max);
    if (!Number.isInteger(target) || target < min || target > max) {
      parallelStatus(`목표는 ${min}~${max} 사이 정수여야 합니다.`, "error");
      input.focus();
      return;
    }
    const scheduler = state.dashboard?.scheduler || {};
    const current = Number(scheduler.desired_simulations ?? scheduler.parallel_target);
    const logicalActive = Number(scheduler.logical_active);
    if (Number.isFinite(current) && target < current && Number.isFinite(logicalActive) && logicalActive > target) {
      const confirmed = window.confirm(
        `목표를 ${current}에서 ${target}(으)로 낮춥니다.\n`
        + "running/attaching 작업은 중단하지 않고, 시작 전으로 확인된 MFT queued 작업만 감소 대상이 됩니다."
      );
      if (!confirmed) return;
    }
    state.updatingParallelTarget = true;
    renderParallelControl(scheduler, state.dashboard?.refill_controller || {});
    let applied = false;
    try {
      const response = await fetch("/api/operator/simulation-policy", {
        method: "PATCH",
        cache: "no-store",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-MFT-Operator-Control": "simulation-policy-v1",
        },
        body: JSON.stringify({
          desired_simulations: target,
          expected_revision: scheduler.policy_revision,
          scale_down_mode: "drain",
        }),
      });
      let payload = {};
      try { payload = await response.json(); } catch (error) { payload = {}; }
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      state.parallelTargetDirty = false;
      if (state.dashboard) state.dashboard.scheduler = { ...scheduler, ...payload, policy_supported: true, connected: true };
      renderParallelControl(state.dashboard?.scheduler || payload, state.dashboard?.refill_controller || {});
      applied = true;
      await refresh();
    } catch (error) {
      parallelStatus(`목표 적용 실패: ${error.message}`, "error");
    } finally {
      state.updatingParallelTarget = false;
      renderParallelControl(state.dashboard?.scheduler || scheduler, state.dashboard?.refill_controller || {});
      if (applied) parallelStatus(`MFT 병렬 목표 ${target}을 저장했습니다. controller가 다음 주기에 맞춥니다.`, "success");
    }
  });
  refresh();
  window.setInterval(refresh, Math.max(5, refreshSeconds) * 1000);
})();
