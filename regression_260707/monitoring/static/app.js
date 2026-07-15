(() => {
  "use strict";

  const hasDOM = typeof document !== "undefined";
  const $ = (selector) => document.querySelector(selector);
  const svgNS = "http://www.w3.org/2000/svg";
  const refreshSeconds = hasDOM ? Number(document.body.dataset.refreshSeconds || 20) : 20;
  const state = {
    dashboard: null, selectedModel: null, selectedModelData: null, refreshing: false,
    historyMetric: "r2",
    parityCache: new Map(), parityRequest: 0,
    parallelTargetDirty: false, updatingParallelTarget: false,
  };

  const labels = {
    loading: "불러오는 중", active: "진행 중", warning: "주의", error: "오류", idle: "대기",
    complete: "완료", waiting: "대기", trained: "학습 완료", stale: "재학습 필요",
    attention: "성능 주의", checkpoint: "체크포인트 CV", not_trained: "학습 전",
    pass: "PASS", fail: "FAIL", unknown: "확인 불가",
  };
  const checkLabels = {
    llt: "Llt 27.5±0.55 µH", temperature: "최고온도 ≤100°C",
    bfield: "설계 B = V/(4fNAe) ≤1.2 T",
    insulation: "절연간격 ≥40 mm", convergence: "수렴오차 ≤1.5%", full_model: "Full model",
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
    setText(
      "#pipeline-exact-rows",
      `${number(cohort.current_strict_full_rows || 0)} · train ${number(cohort.first_training_rows || 500)} / model ${number(cohort.model_activation_rows || 3000)} / tune ${number(cohort.first_tuning_rows || 4000)}`,
    );
    renderPipelineRevision(
      "#pipeline-dataset-generation",
      cohort.generation,
      cohort.available === true,
    );
    const external = pipeline.external_tuners || {};
    const externalProcesses = Array.isArray(external.processes) ? external.processes : [];
    const externalState = $("#pipeline-external-tuner-state");
    externalState.className = `state-chip ${externalProcesses.length ? "unknown" : external.available ? "pass" : "fail"}`;
    externalState.textContent = externalProcesses.length ? `${externalProcesses.length}개 별도 실행` : external.available ? "없음" : "확인 불가";
    setText(
      "#pipeline-external-tuner-detail",
      externalProcesses.length
        ? "아래 작업은 durable lane 수에 포함되지 않습니다."
        : external.error || "durable queue 밖의 Optuna 작업이 없습니다.",
    );
    const externalList = $("#pipeline-external-tuners");
    externalList.replaceChildren();
    externalProcesses.forEach((process) => {
      const dataset = compactGeneration(process.dataset);
      externalList.append(element(
        "li", "",
        `PID ${process.pid} · ${duration(process.elapsed_seconds)} · trials ${process.trials || "—"} · ${dataset}`,
      ));
    });

    const parallel = pipeline.parallel || {};
    const queue = pipeline.queue || {};
    const counts = queue.counts || {};
    setText("#pipeline-running-lanes", `${number(parallel.running_lane_count || 0)} / 6`);
    setText("#pipeline-active-lanes", `${number(parallel.active_lane_count || 0)} / 6`);
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
    setText("#data-total", `${number(data.total_rows)}개`);
    const physicsRevision = String(data.current_physics_data_revision || "");
    const physicsLabel = physicsRevision.length > 24 ? `${physicsRevision.slice(0, 24)}…` : (physicsRevision || "—");
    setText("#data-strict-detail", `physics ${physicsLabel} · strict full`);
    const memberShas = Array.isArray(data.member_git_hash_shorts) ? data.member_git_hash_shorts : [];
    setText("#data-member-shas", `SHA: ${memberShas.length ? memberShas.join(", ") : "—"}`);
    setText("#data-raw-total", `${number(data.raw_total_rows)}개`);
    setText("#data-quality-detail", `현재 revision raw ${number(data.revision_raw_rows)}개 · strict EM ${number(data.em_valid_rows)}개 · strict full ${number(data.total_rows)}개`);
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

  function renderParity(model, payload = {}) {
    const svg = $("#model-parity-chart");
    svg.replaceChildren();
    const rawPairs = Array.isArray(payload.pairs)
      ? payload.pairs
      : Array.isArray(payload.parity) ? payload.parity : [];
    const points = rawPairs
      .map((item) => ({ actual: Number(item.actual), predicted: Number(item.predicted) }))
      .filter((item) => Number.isFinite(item.actual) && Number.isFinite(item.predicted));
    const empty = $("#model-parity-empty");
    empty.classList.toggle("hidden", points.length > 0);
    if (!points.length) {
      empty.textContent = payload.error || "검증된 parity 데이터가 없습니다.";
      setText("#model-parity-meta", model.evaluated ? `checkpoint ${model.checkpoint ?? "—"}` : "—");
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
    setText("#model-parity-meta", `${prefix} · ${number(sampleCount)}/${number(totalCount)}점 · R² ${number(model.r2, 3)}`);
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
    const cacheKey = [model.target, model.checkpoint, model.evaluated_at, model.parity_sample_count].join(":");
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

  function renderNsga(payload) {
    setText("#nsga-status", payload.available ? `${payload.status === "running" ? "실행 중" : "완료"} · AL ${payload.al_stage || "—"}` : "실행 전");
    setText("#nsga-round", payload.round == null ? "—" : `#${String(payload.round).padStart(2, "0")}`);
    setText("#nsga-count", payload.available ? `${number(payload.candidate_count)}개` : "—");
    setText("#nsga-min-volume", compact(payload.summary?.min_volume_L, "L"));
    setText("#nsga-min-loss", compact(payload.summary?.min_loss_W, "W"));
    const comparison = payload.comparison;
    setText("#nsga-comparison", comparison?.min_volume_change_L == null ? "이전 round 비교 없음" : `이전 대비 최소체적 ${comparison.min_volume_change_L > 0 ? "+" : ""}${number(comparison.min_volume_change_L, 1)} L`);
    setText("#nsga-note", payload.note || "");
    const candidates = payload.candidates || [];
    renderScatter(candidates);
    $("#nsga-empty").classList.toggle("hidden", candidates.length > 0);
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

  function renderScatter(candidates) {
    const svg = $("#nsga-chart"); svg.replaceChildren();
    const points = candidates.filter((item) => Number.isFinite(item.volume_L) && Number.isFinite(item.total_loss_W));
    if (!points.length) return;
    const box = chartBounds(svg); box.bottom = 38;
    let xMin = Math.min(...points.map((item) => item.volume_L)), xMax = Math.max(...points.map((item) => item.volume_L));
    let yMin = Math.min(...points.map((item) => item.total_loss_W)), yMax = Math.max(...points.map((item) => item.total_loss_W));
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
    points.forEach((candidate) => {
      const circle = svgElement("circle", { cx: sx(candidate.volume_L), cy: sy(candidate.total_loss_W), r: candidate.is_min_volume ? 5 : 3.2, class: `point${candidate.is_min_volume ? " highlight" : ""}`, tabindex: 0 });
      const title = svgElement("title"); title.textContent = `${candidate.id} · ${number(candidate.volume_L, 1)} L · ${number(candidate.total_loss_W, 1)} W`; circle.append(title);
      circle.addEventListener("click", () => showCandidate(candidate));
      circle.addEventListener("keydown", (event) => { if (event.key === "Enter") showCandidate(candidate); });
      svg.append(circle);
    });
  }

  function makeChecks(container, checks) {
    container.replaceChildren();
    Object.entries(checks || {}).forEach(([key, check]) => {
      const status = check.pass === true ? "pass" : check.pass === false ? "fail" : "unknown";
      const item = element("div", `check-item ${status}`);
      item.append(element("span", "", checkLabels[key] || key));
      const value = check.value == null ? "확인 불가" : `${number(check.value, Math.abs(check.value) < 10 ? 3 : 1)} · ${labels[status]}`;
      item.append(element("b", "", value)); container.append(item);
    });
  }

  function showCandidate(candidate) {
    setText("#dialog-title", candidate.id);
    const report = candidate.report || {};
    const summary = $("#dialog-summary"); summary.replaceChildren();
    const turns = [report.turns_primary, report.turns_secondary_center, report.turns_secondary_side]
      .every((value) => Number.isFinite(Number(value)))
      ? `${number(report.turns_primary)} / ${number(report.turns_secondary_center)} / ${number(report.turns_secondary_side)}`
      : "—";
    const leakage = Number.isFinite(Number(report.leakage_target_uH))
      ? `${compact(candidate.pred_Llt_phys, "µH")} / 목표 ${compact(report.leakage_target_uH, "µH")}`
      : compact(candidate.pred_Llt_phys, "µH");
    [
      ["크기 (W×L×H)", report.size_WxLxH_mm ? `${report.size_WxLxH_mm} mm` : "—"],
      ["체적", compact(candidate.volume_L, "L")],
      ["바닥면적", compact(report.footprint_cm2, "cm²")],
      ["턴수 (1차 / 2차 중앙 / 측면)", turns],
      ["누설 인덕턴스", leakage],
      ["설계 자속밀도", compact(candidate.B_design_analytic_T, "T")],
      ["총손실", compact(candidate.total_loss_W, "W")],
      ["효율", compact(report.pred_efficiency_pct, "%")],
    ].forEach(([name, value]) => {
      const box = element("div"); box.append(element("span", "", name), element("b", "", value)); summary.append(box);
    });
    makeChecks($("#dialog-checks"), candidate.constraints);

    const appendDetails = (selector, items) => {
      const container = $(selector); container.replaceChildren();
      items.forEach(([label, value]) => {
        const item = element("div");
        item.append(element("dt", "", label), element("dd", "", value));
        container.append(item);
      });
    };
    appendDetails("#dialog-performance", [
      ["코어 손실", compact(report.pred_core_loss_W, "W")],
      ["1차 권선 손실", compact(report.pred_primary_winding_loss_W, "W")],
      ["2차 중앙 권선 손실", compact(report.pred_secondary_center_winding_loss_W, "W")],
      ["2차 측면 권선 손실", compact(report.pred_secondary_side_winding_loss_W, "W")],
      ["2차 권선 손실 합계", compact(report.pred_secondary_winding_loss_W, "W")],
      ["전체 권선 손실", compact(report.pred_total_winding_loss_W, "W")],
      ["코어 콜드플레이트 손실", compact(report.pred_core_cold_plate_loss_W, "W")],
      ["권선 콜드플레이트 손실", compact(report.pred_winding_cold_plate_loss_W, "W")],
      ["정격 출력", compact(report.rated_power_W, "W")],
      ["FEA 평균 B surrogate (참고)", compact(candidate.pred_B_mean_core, "T")],
    ]);
    appendDetails("#dialog-derived", [
      ["1차 권선 1턴 두께", compact(report.cw1_conductor_thickness_mm, "mm")],
      ["2차 권선 1턴 두께", compact(report.cw2_conductor_thickness_mm, "mm")],
      ["1차 / 2차 턴 간격", `${compact(report.gap1_mm, "mm")} / ${compact(report.gap2_mm, "mm")}`],
      ["1차 중앙 권선팩 폭", compact(report.nwl1_main_pack_width_mm, "mm")],
      ["2차 중앙 권선팩 폭", compact(report.nwl2_main_pack_width_mm, "mm")],
      ["2차 측면 권선팩 폭", compact(report.nwl2_side_pack_width_mm, "mm")],
      ["1차 / 2차 권선 높이", `${compact(report.nwh1_winding_height_mm, "mm")} / ${compact(report.nwh2_winding_height_mm, "mm")}`],
      ["코어 조 수 / 1조 깊이", `${number(report.n_core_group)} / ${compact(report.core_depth_each_mm, "mm")}`],
      ["코어 콜드플레이트 / 패드", `${compact(report.core_cold_plate_thickness_mm, "mm")} / ${compact(report.core_thermal_pad_thickness_mm, "mm")}`],
      ["권선 콜드플레이트 / 패드", `${compact(report.winding_cold_plate_thickness_mm, "mm")} / ${compact(report.winding_thermal_pad_thickness_mm, "mm")}`],
      ["권선 콜드플레이트 길이", `${compact(report.wcp_len_pct, "%")} / ${compact(report.wcp_len_x_mm, "mm")}`],
      ["코어 유효 단면적", compact(report.Ae_effective_m2, "m²")],
      ["적층계수", compact(report.core_lamination_factor)],
      ["초기 0.7식 B (감사용)", compact(report.B_legacy_0p7_T, "T")],
    ]);
    const params = $("#dialog-parameters"); params.replaceChildren();
    const parameterUnits = {
      l1: "mm", l2: "mm", h1: "mm", w1: "mm", core_plate_t: "mm",
      core_plate_pad_t: "mm", cw1: "mm", gap1: "mm", cw2: "mm", gap2: "mm",
      nwh1: "mm", nwh2: "mm", wcp_t: "mm", wcp_pad_t: "mm",
      wcp_len_pct: "%", wcp_len_x: "mm",
    };
    Object.entries(candidate.parameters || {}).forEach(([key, value]) => {
      const rendered = number(value, Number.isInteger(value) ? 0 : 3);
      const unit = parameterUnits[key] || "";
      const item = element("div"); item.append(element("dt", "", key), element("dd", "", `${rendered}${unit ? ` ${unit}` : ""}`)); params.append(item);
    });
    $("#candidate-dialog").showModal();
  }

  function renderVerification(payload) {
    setText("#verify-stage", payload.stage === "NOT_STARTED" ? "검증 전" : `${payload.stage} · round ${payload.round ?? "—"}`);
    const counts = payload.counts || {};
    setText("#verify-coverage", counts.total ? `유효 ${number(counts.valid)} / ${number(counts.total)} · coverage ${counts.coverage == null ? "—" : `${number(counts.coverage * 100, 1)}%`}` : "아직 제출된 검증이 없습니다.");
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
    setText("#parallel-control-eyebrow", policySupported ? "DURABLE SIMULATION POLICY · MFT ONLY" : "AUTOMATIC REFILL CONTROL · MFT ONLY");
    setText("#parallel-control-title", policySupported ? "MFT 병렬 실행 목표" : "MFT 자동 실행 유지");
    setText(
      "#parallel-control-description",
      policySupported
        ? "attaching + active 수를 desired 목표로 유지합니다. queued admission은 별도로 표시하며 IPMSM에는 적용하지 않습니다."
        : "외부 refill-controller가 queued + attaching + running 합계를 자동으로 유지합니다.",
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
    renderParallelControl(payload.scheduler || {}, payload.refill_controller || {});
    renderData(payload.data || {});
    renderModels(payload.models || { models: [] });
    renderNsga(payload.nsga2 || { candidates: [] });
    renderVerification(payload.verification || { counts: {}, final: {} });
    renderDiagnostics(payload);
  }

  async function refresh() {
    if (state.refreshing) return;
    state.refreshing = true;
    $("#refresh-button").disabled = true;
    $("#loading-indicator").classList.add("loading");
    try {
      const response = await fetch("/api/dashboard", { cache: "no-store", headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (payload.error || !payload.data) throw new Error(payload.error || "dashboard payload is incomplete");
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
    return;
  }

  $("#refresh-button").addEventListener("click", refresh);
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
