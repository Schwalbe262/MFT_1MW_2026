(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const svgNS = "http://www.w3.org/2000/svg";
  const refreshSeconds = Number(document.body.dataset.refreshSeconds || 20);
  const state = { dashboard: null, selectedModel: null, refreshing: false };

  const labels = {
    loading: "불러오는 중", active: "진행 중", warning: "주의", error: "오류", idle: "대기",
    complete: "완료", waiting: "대기", trained: "학습 완료", stale: "재학습 필요",
    attention: "성능 주의", not_trained: "학습 전", pass: "PASS", fail: "FAIL", unknown: "확인 불가",
  };
  const checkLabels = {
    llt: "Llt 27.5±0.55 µH", temperature: "최고온도 ≤100°C", bmax: "Bmax ≤1.2 T",
    insulation: "절연간격 ≥40 mm", convergence: "수렴오차 ≤1.5%", full_model: "Full model",
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

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
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

  function renderData(data) {
    setText("#data-total", `${number(data.total_rows)}개`);
    setText("#data-complete", `${number(data.complete_rows)}개`);
    setText("#data-quality-detail", `EM 유효 ${number(data.em_valid_rows)} · Thermal 유효 ${number(data.thermal_valid_rows)}`);
    setText("#data-throughput", `+${number(data.throughput_1h)}개`);
    setText("#data-throughput-detail", `24시간 +${number(data.added_24h)} · 유효속도 ${number(data.effective_hourly_rate, 1)}/h`);
    setText("#data-eta", data.eta_3000 ? dateTime(data.eta_3000) : "산정 불가");
    setText("#data-stall", data.eta_hours != null ? `약 ${number(data.eta_hours, 1)}시간 후` : "최근 처리량이 없습니다");
    setText("#data-freshness", elapsed(data.stalled_minutes));
    $("#data-freshness").classList.toggle("stale", Boolean(data.stalled));
    setText("#latest-revision", data.latest_revision ? `revision ${data.latest_revision.slice(0, 10)}` : "revision —");
    setText("#goal-label", `${number(data.total_rows)} / ${number(data.goal)}`);
    setText("#stretch-label", `${number(data.total_rows)} / ${number(data.stretch_goal)}+`);
    $("#goal-progress").style.width = `${Math.max(0, Math.min(100, data.goal_progress_pct || 0))}%`;
    $("#stretch-progress").style.width = `${Math.max(0, Math.min(100, data.stretch_progress_pct || 0))}%`;
    setText("#data-24h", `+${number(data.added_24h)}개`);
    setText("#data-remaining", `${number(data.remaining_to_goal)}개`);
    setText("#revision-mismatch", data.rows_not_latest_revision == null ? "—" : `${number(data.rows_not_latest_revision)}개`);
    setText("#collector-nodata", data.collector?.no_data_tasks == null ? "—" : `${number(data.collector.no_data_tasks)}건`);
    lineChart($("#data-chart"), data.history || [], {
      x: (item) => new Date(item.time).getTime(), y: (item) => Number(item.total),
      xLabel: (value) => dateTime(new Date(value).toISOString()), yLabel: (value) => number(value),
      color: "#26d7c7", area: true,
    });
    $("#data-chart-empty").classList.toggle("hidden", (data.history || []).length > 0);
  }

  function chartBounds(svg) {
    const viewBox = svg.viewBox.baseVal;
    return { width: viewBox.width || 760, height: viewBox.height || 260, left: 52, right: 18, top: 18, bottom: 34 };
  }

  function lineChart(svg, values, options) {
    svg.replaceChildren();
    if (!values.length) return;
    const box = chartBounds(svg);
    const xs = values.map(options.x).filter(Number.isFinite);
    const ys = values.map(options.y).filter(Number.isFinite);
    if (!xs.length || !ys.length) return;
    let xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    yMin = Math.min(0, yMin);
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
      const text = svgElement("text", { x: sx(value), y: box.height - 9, "text-anchor": ratio === 0 ? "start" : ratio === 1 ? "end" : "middle" });
      text.textContent = options.xLabel(value); svg.append(text);
    });
    const points = values.map((item) => [sx(options.x(item)), sy(options.y(item))]).filter((point) => point.every(Number.isFinite));
    if (options.area && points.length) {
      const path = `M ${points[0][0]} ${box.height - box.bottom} L ${points.map((point) => point.join(" ")).join(" L ")} L ${points.at(-1)[0]} ${box.height - box.bottom} Z`;
      svg.append(svgElement("path", { d: path, fill: `url(#area-gradient-${svg.id})` }));
    }
    svg.append(svgElement("path", { d: `M ${points.map((point) => point.join(" ")).join(" L ")}`, class: "data-line", stroke: options.color || "#26d7c7" }));
    points.forEach(([x, y], index) => {
      if (points.length <= 40 || index === points.length - 1) svg.append(svgElement("circle", { cx: x, cy: y, r: 2.8, fill: options.color || "#26d7c7" }));
    });
  }

  function renderModels(payload) {
    setText("#model-summary", `${number(payload.trained_count)} / ${number(payload.target_count)} 학습`);
    setText("#model-quality-note", payload.quality_note || "");
    const tbody = $("#model-table");
    tbody.replaceChildren();
    (payload.models || []).forEach((model) => {
      const row = element("tr"); row.dataset.clickable = "true"; row.dataset.target = model.target;
      const nameCell = element("td");
      const name = element("div", "model-name"); name.append(element("b", "", model.label), element("code", "", model.target)); nameCell.append(name);
      row.append(nameCell);
      row.append(element("td", "table-value", model.trained ? `${number(model.n_train)}/${number(model.n_holdout)}` : "—"));
      const r2 = element("td", "table-value", model.r2 == null ? "—" : number(model.r2, 3));
      if (model.delta_r2 != null) r2.append(element("span", model.delta_r2 >= 0 ? "delta-up" : "delta-down", ` ${model.delta_r2 >= 0 ? "▲" : "▼"}${number(Math.abs(model.delta_r2), 3)}`));
      row.append(r2);
      row.append(element("td", "table-value", compact(model.rmse, model.unit)));
      row.append(element("td", "table-value", model.mape_pct == null ? "—" : `${number(model.mape_pct, 2)}%`));
      row.append(element("td", "table-value", model.p90_ape_pct == null ? "—" : `${number(model.p90_ape_pct, 2)}%`));
      const statusCell = element("td"); statusCell.append(element("span", `state-chip ${model.status}`, labels[model.status] || model.status)); row.append(statusCell);
      row.addEventListener("click", () => selectModel(model));
      tbody.append(row);
    });
    let selected = (payload.models || []).find((model) => model.target === state.selectedModel);
    if (!selected) selected = (payload.models || []).find((model) => model.trained) || payload.models?.[0];
    if (selected) selectModel(selected);
  }

  function selectModel(model) {
    state.selectedModel = model.target;
    document.querySelectorAll("#model-table tr").forEach((row) => row.classList.toggle("selected", row.dataset.target === model.target));
    setText("#model-chart-title", model.label);
    setText("#model-trained-at", model.trained_at ? `학습 ${dateTime(model.trained_at)}` : "학습 전");
    const history = model.history || [];
    $("#model-chart-empty").classList.toggle("hidden", history.length > 0);
    lineChart($("#model-chart"), history, {
      x: (item) => new Date(item.time).getTime(), y: (item) => Number(item.r2),
      xLabel: (value) => dateTime(new Date(value).toISOString()), yLabel: (value) => number(value, 2),
      color: "#55aaff", area: true,
    });
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
      row.append(element("td", "table-value", compact(candidate.pred_B_max_core, "T")));
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
    const summary = $("#dialog-summary"); summary.replaceChildren();
    [["체적", compact(candidate.volume_L, "L")], ["총손실", compact(candidate.total_loss_W, "W")], ["Llt", compact(candidate.pred_Llt_phys, "µH")], ["Bmax", compact(candidate.pred_B_max_core, "T")]].forEach(([name, value]) => {
      const box = element("div"); box.append(element("span", "", name), element("b", "", value)); summary.append(box);
    });
    makeChecks($("#dialog-checks"), candidate.constraints);
    const params = $("#dialog-parameters"); params.replaceChildren();
    const parameterUnits = { wcp_len_pct: "%", wcp_len_x: "mm" };
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
    setText("#final-error", final.error);
    $("#final-error").classList.toggle("hidden", !final.error);
  }

  function renderDiagnostics(payload) {
    const scheduler = payload.scheduler || {};
    const details = $("#scheduler-details"); details.replaceChildren();
    [["연결", scheduler.connected ? "정상" : "실패"], ["실행 / 대기", `${number(scheduler.running)} / ${number(scheduler.pending)}`], ["완료 / 실패", `${number(scheduler.completed)} / ${number(scheduler.failed)}`], ["조회 범위", scheduler.task_prefix || "—"], ["모드", "GET only"]].forEach(([key, value]) => {
      const item = element("div"); item.append(element("dt", "", key), element("dd", "", value)); details.append(item);
    });
    const warnings = [
      ...(payload.data?.warnings || []), ...(payload.models?.warnings || []),
      ...(payload.nsga2?.warnings || []), ...(payload.verification?.warnings || []),
      ...(scheduler.error ? [scheduler.error] : []),
    ];
    const list = $("#artifact-warnings"); list.replaceChildren();
    [...new Set(warnings)].forEach((warning) => list.append(element("li", "", warning)));
    if (!warnings.length) list.append(element("li", "", "없음"));
  }

  function render(payload) {
    state.dashboard = payload;
    renderOverall(payload);
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

  $("#refresh-button").addEventListener("click", refresh);
  refresh();
  window.setInterval(refresh, Math.max(5, refreshSeconds) * 1000);
})();
