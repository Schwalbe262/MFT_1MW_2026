(() => {
  "use strict";

  const LEGACY_REVISION = "legacy_unspecified";
  function hasNumber(value) {
    return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  }

  function number(value, digits = 0) {
    if (!hasNumber(value)) return "—";
    return Number(value).toLocaleString("ko-KR", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function isActive(cohort) {
    return Boolean(cohort?.active ?? cohort?.current);
  }

  function cohortTimestamp(cohort) {
    const timestamp = String(cohort?.latest_saved_at || "");
    const fractional = timestamp.match(/\.(\d+)(?:Z|[+-]\d{2}:?\d{2})?$/)?.[1] || "";
    const normalized = fractional
      ? timestamp.replace(`.${fractional}`, `.${fractional.padEnd(3, "0").slice(0, 3)}`)
      : timestamp;
    const milliseconds = Date.parse(normalized);
    if (!Number.isFinite(milliseconds)) return Number.NEGATIVE_INFINITY;
    return milliseconds + Number(fractional.padEnd(6, "0").slice(3, 6) || 0) / 1000;
  }

  function isZeroLegacyNoise(cohort) {
    return !isActive(cohort)
      && String(cohort?.physics_data_revision || "").trim().toLowerCase() === LEGACY_REVISION
      && Number(cohort?.strict_em_rows) === 0
      && Number(cohort?.strict_full_rows) === 0
      && (!hasNumber(cohort?.growth_rate_per_hour) || Number(cohort.growth_rate_per_hour) === 0);
  }

  function compactCohorts(payload) {
    const cohorts = Array.isArray(payload) ? payload.filter((item) => item && typeof item === "object") : [];
    const legacy = cohorts.filter(isZeroLegacyNoise);
    const rows = cohorts.filter((cohort) => !isZeroLegacyNoise(cohort));
    if (legacy.length) {
      const sum = (key) => legacy.reduce(
        (total, cohort) => total + (hasNumber(cohort?.[key]) ? Number(cohort[key]) : 0),
        0,
      );
      const newest = [...legacy].sort((left, right) => cohortTimestamp(right) - cohortTimestamp(left))[0];
      rows.push({
        git_hash: null,
        git_hash_short: "legacy",
        physics_data_revision: LEGACY_REVISION,
        latest_saved_at: newest?.latest_saved_at || null,
        active: false,
        legacy_aggregate: true,
        cohort_count: legacy.length,
        raw_rows: sum("raw_rows"),
        strict_em_rows: sum("strict_em_rows"),
        strict_full_rows: sum("strict_full_rows"),
        growth_rate_per_hour: sum("growth_rate_per_hour"),
      });
    }
    return rows.sort((left, right) => {
      if (isActive(left) !== isActive(right)) return isActive(left) ? -1 : 1;
      const newestFirst = cohortTimestamp(right) - cohortTimestamp(left);
      if (newestFirst) return newestFirst;
      const leftKey = `${left?.git_hash_short || left?.git_hash || ""}:${left?.physics_data_revision || ""}`;
      const rightKey = `${right?.git_hash_short || right?.git_hash || ""}:${right?.physics_data_revision || ""}`;
      return leftKey.localeCompare(rightKey);
    });
  }

  function shortRevision(value) {
    const revision = String(value || "").trim();
    if (!revision || revision.toLowerCase() === LEGACY_REVISION) return "legacy";
    return revision.length > 22 ? `${revision.slice(0, 22)}…` : revision;
  }

  function growth(cohort) {
    if (!hasNumber(cohort?.growth_rate_per_hour)) return "—";
    const value = Number(cohort.growth_rate_per_hour);
    return `${value > 0 ? "+" : ""}${number(value, 1)}/h`;
  }

  function element(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== "") node.textContent = text;
    return node;
  }

  function setStatus(kind, text) {
    const status = document.querySelector("#cohorts-status");
    status.className = `status-pill status-${kind}`;
    status.innerHTML = "";
    status.append(element("i"), document.createTextNode(` ${text}`));
  }

  function render(data) {
    const rows = compactCohorts(data?.cohorts);
    const body = document.querySelector("#cohorts-body");
    body.replaceChildren();
    rows.forEach((cohort) => {
      const row = element("tr", isActive(cohort) ? "active" : (cohort.legacy_aggregate ? "legacy-aggregate" : ""));
      const sha = cohort.legacy_aggregate
        ? `레거시 (${number(cohort.cohort_count)}개 코호트)`
        : cohort.git_hash_short || (cohort.git_hash ? String(cohort.git_hash).slice(0, 10) : "—");
      const shaCell = element("td", "cohort-history-sha mono", sha);
      if (cohort.git_hash) shaCell.title = String(cohort.git_hash);
      if (isActive(cohort)) shaCell.append(element("span", "cohort-active-tag", "ACTIVE"));
      const revisionCell = element("td", "cohort-history-revision mono", shortRevision(cohort.physics_data_revision));
      if (cohort.physics_data_revision) revisionCell.title = String(cohort.physics_data_revision);
      const growthCell = element("td", "cohort-history-growth", growth(cohort));
      if (Number(cohort.growth_rate_per_hour) > 0) growthCell.classList.add("positive");
      row.append(
        shaCell,
        revisionCell,
        element("td", "", number(cohort.raw_rows)),
        element("td", "", number(cohort.strict_em_rows)),
        element("td", "", number(cohort.strict_full_rows)),
        growthCell,
      );
      body.append(row);
    });
    document.querySelector("#cohorts-empty").classList.toggle("hidden", rows.length > 0);
    const active = rows.find(isActive);
    document.querySelector("#cohorts-summary").textContent = active
      ? `${number(rows.length)}개 행 · ACTIVE ${active.git_hash_short || "—"}`
      : `${number(rows.length)}개 행`;
    const revision = data?.current_physics_data_revision;
    document.querySelector("#cohorts-current-revision").textContent = revision
      ? `현재 physics ${shortRevision(revision)}`
      : "현재 physics —";
    const members = Array.isArray(data?.member_git_hash_shorts) ? data.member_git_hash_shorts : [];
    document.querySelector("#cohorts-members").textContent = `SHA: ${members.length ? members.join(", ") : "—"}`;
  }

  async function refresh() {
    const button = document.querySelector("#cohorts-refresh");
    button.disabled = true;
    try {
      const response = await fetch("/api/data", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data?.available === false && data?.error) throw new Error(data.error);
      render(data);
      setStatus("active", "최신");
      document.querySelector("#cohorts-refreshed").textContent = new Date().toLocaleTimeString("ko-KR");
      document.querySelector("#cohorts-error").classList.add("hidden");
    } catch (error) {
      setStatus("warning", "확인 필요");
      const message = document.querySelector("#cohorts-error");
      message.textContent = `코호트 데이터를 불러오지 못했습니다: ${error?.message || error}`;
      message.classList.remove("hidden");
    } finally {
      button.disabled = false;
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { compactCohorts };
  }
  if (typeof document !== "undefined") {
    document.querySelector("#cohorts-refresh").addEventListener("click", refresh);
    refresh();
    const seconds = Math.max(5, Number(document.body.dataset.refreshSeconds || 20));
    window.setInterval(refresh, seconds * 1000);
  }
})();
