"""Safe readers and view-model builders for MFT campaign artifacts.

This module intentionally depends only on the Python standard library.  The
simulation/training environments therefore do not need to be imported by the
web process, and a partially written or corrupt artifact cannot take down the
dashboard.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
DATA_GOAL = 3_000
STRETCH_GOAL = 10_000
TARGETS: tuple[dict[str, str], ...] = (
    {"name": "Llt_phys", "label": "누설 인덕턴스 (Llt)", "unit": "µH"},
    {"name": "P_winding_total", "label": "권선 손실", "unit": "W"},
    {"name": "P_core_total", "label": "코어 손실", "unit": "W"},
    {"name": "P_core_plate_total", "label": "코어 플레이트 손실", "unit": "W"},
    {"name": "P_wcp_total", "label": "권선 냉각판 손실", "unit": "W"},
    {"name": "B_max_core", "label": "코어 최대 자속밀도", "unit": "T"},
    {"name": "Tprobe_Tx_leeward_max", "label": "Tx 최대 온도", "unit": "°C"},
    {"name": "Tprobe_Rx_main_leeward_max", "label": "Rx main 최대 온도", "unit": "°C"},
    {"name": "Tprobe_Rx_side_leeward_max", "label": "Rx side 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_center_max", "label": "코어 최대 온도", "unit": "°C"},
    {"name": "k", "label": "결합계수 (k)", "unit": ""},
    {"name": "B_mean_core", "label": "코어 평균 자속밀도", "unit": "T"},
)
TARGET_META = {item["name"]: item for item in TARGETS}
TEMPERATURE_TARGETS = (
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_core_center_max",
)
DESIGN_PARAMETER_KEYS = (
    "N1_main", "N2_main", "N2_side", "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "cw1", "gap1", "cw2", "gap2",
    "nwh1", "nwh2", "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y", "w1c_w2s_gap_x_actual",
    "w1s_cs_space_x", "cs_w1s_space_y", "h_gap2", "wcp_t", "wcp_len_x",
)
INSULATION_KEYS = (
    "cc_w2c_space_x", "cc_w2c_space_y", "w2c_w1c_space_x",
    "w2c_w1c_space_y", "w1c_w2s_gap_x_actual", "w1s_cs_space_x",
    "cs_w1s_space_y", "h_gap2",
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, default: int = 0) -> int:
    number = _finite_number(value)
    return int(number) if number is not None else default


def _flag(value: Any) -> bool:
    number = _finite_number(value)
    if number is not None:
        return number == 1.0
    return str(value).strip().lower() in {"true", "yes", "pass", "passed"}


def _safe_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _parse_time(value: Any, local_tz=None) -> datetime | None:
    text = _safe_text(value, 80)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%y%m%d_%H%M%S_%f", "%y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz or _now().tzinfo)
    return parsed


def _coerce(value: Any) -> Any:
    number = _finite_number(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return _safe_text(value)


@dataclass(frozen=True)
class ReadResult:
    value: Any
    path: str
    exists: bool
    mtime: datetime | None = None
    warning: str | None = None


@dataclass
class _CacheEntry:
    signature: tuple[int, int]
    value: Any
    mtime: datetime


class SafeArtifactCache:
    """Caches the last good parse of each artifact by size and mtime."""

    def __init__(self) -> None:
        self._good: dict[tuple[str, str], _CacheEntry] = {}
        self._failed: dict[tuple[str, str], tuple[int, int]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _read(
        self,
        path: Path,
        kind: str,
        parser: Callable[[Path], Any],
        default: Any,
    ) -> ReadResult:
        key = (str(path.resolve()), kind)
        try:
            signature = self._signature(path)
        except FileNotFoundError:
            return ReadResult(default, str(path), False)
        except OSError as exc:
            return ReadResult(default, str(path), False, warning=f"{path.name} 상태 확인 실패: {exc}")

        with self._lock:
            cached = self._good.get(key)
            if cached and cached.signature == signature:
                return ReadResult(cached.value, str(path), True, cached.mtime)
            if self._failed.get(key) == signature:
                previous = cached.value if cached else default
                previous_time = cached.mtime if cached else None
                return ReadResult(
                    previous,
                    str(path),
                    True,
                    previous_time,
                    f"{path.name} 손상/작성 중: 마지막 정상 데이터를 표시합니다.",
                )

        try:
            value = parser(path)
            mtime = datetime.fromtimestamp(signature[0] / 1_000_000_000, tz=_now().tzinfo)
        except (OSError, UnicodeError, ValueError, TypeError, csv.Error, json.JSONDecodeError) as exc:
            with self._lock:
                self._failed[key] = signature
                cached = self._good.get(key)
            previous = cached.value if cached else default
            previous_time = cached.mtime if cached else None
            return ReadResult(
                previous,
                str(path),
                True,
                previous_time,
                f"{path.name} 읽기 실패: {type(exc).__name__}: {exc}",
            )

        with self._lock:
            self._good[key] = _CacheEntry(signature, value, mtime)
            self._failed.pop(key, None)
        return ReadResult(value, str(path), True, mtime)

    def json(self, path: Path, default: Any = None, max_bytes: int = 16 * 1024 * 1024) -> ReadResult:
        def parser(source: Path) -> Any:
            if source.stat().st_size > max_bytes:
                raise ValueError(f"file exceeds {max_bytes} byte safety limit")
            with source.open("r", encoding="utf-8-sig") as handle:
                return json.load(handle)

        return self._read(path, "json", parser, default)

    def csv(self, path: Path, max_rows: int = 100_000) -> ReadResult:
        def parser(source: Path) -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("CSV header is missing")
                for index, row in enumerate(reader):
                    if index >= max_rows:
                        raise ValueError(f"CSV exceeds {max_rows} row safety limit")
                    rows.append(dict(row))
            return rows

        return self._read(path, f"csv:{max_rows}", parser, [],)


class SchedulerReader:
    """GET-only adapter for the existing scheduler API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        task_prefix: str = "mft",
        timeout: float = 2.0,
        ttl: float = 10.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_prefix = task_prefix
        self.timeout = timeout
        self.ttl = ttl
        self._opener = opener
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with self._lock:
            if self._cached is not None and now_monotonic - self._cached_at < self.ttl:
                return self._cached
        # The campaign has more than ten thousand historical tasks.  Reading
        # /api/tasks would make a dashboard refresh take many seconds, so use
        # the scheduler's aggregate, read-only endpoint exclusively.
        query = urlencode({"name_prefix": self.task_prefix})
        url = f"{self.base_url}/api/tasks/summary?{query}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "mft-monitor/1"}, method="GET")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("statuses"), dict):
                raise ValueError("scheduler summary response is invalid")
            statuses = Counter({
                str(name).strip().lower(): _integer(count, 0)
                for name, count in payload["statuses"].items()
            })
            running = sum(statuses.get(name, 0) for name in ("running", "executing"))
            pending = sum(statuses.get(name, 0) for name in ("pending", "queued", "submitted"))
            completed = sum(statuses.get(name, 0) for name in ("completed", "complete", "succeeded", "success"))
            failed = sum(statuses.get(name, 0) for name in ("failed", "error", "timed_out", "timeout"))
            cancelled = sum(statuses.get(name, 0) for name in ("cancelled", "canceled"))
            result = {
                "connected": True,
                "url": self.base_url,
                "read_only": True,
                "task_prefix": self.task_prefix,
                "total": _integer(payload.get("total"), sum(statuses.values())),
                "running": running,
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "other": max(0, _integer(payload.get("total"), sum(statuses.values())) - running - pending - completed - failed - cancelled),
                "statuses": dict(sorted(statuses.items())),
                "error": None,
                "updated_at": _iso(_now()),
            }
        except (HTTPError, URLError, OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            result = {
                "connected": False,
                "url": self.base_url,
                "read_only": True,
                "task_prefix": self.task_prefix,
                "total": 0,
                "running": 0,
                "pending": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "other": 0,
                "statuses": {},
                "error": f"scheduler 조회 실패: {type(exc).__name__}: {exc}",
                "updated_at": _iso(_now()),
            }
        with self._lock:
            self._cached = result
            self._cached_at = now_monotonic
        return result


class RuntimeRecorder:
    """Persists a compact current snapshot and low-frequency history."""

    def __init__(self, directory: Path, min_interval_seconds: int = 60) -> None:
        self.directory = directory
        self.snapshot_path = directory / "monitor_snapshot.json"
        self.history_path = directory / "monitor_history.jsonl"
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._last_signature: tuple[Any, ...] | None = None
        self._last_write = 0.0

    @staticmethod
    def _summary(dashboard: dict[str, Any]) -> dict[str, Any]:
        data = dashboard.get("data", {})
        models = dashboard.get("models", {})
        nsga = dashboard.get("nsga2", {})
        verification = dashboard.get("verification", {})
        scheduler = dashboard.get("scheduler", {})
        return {
            "schema_version": SCHEMA_VERSION,
            "time": dashboard.get("generated_at"),
            "overall": dashboard.get("status", {}).get("overall"),
            "data": {
                "total_rows": data.get("total_rows"),
                "complete_rows": data.get("complete_rows"),
                "throughput_1h": data.get("throughput_1h"),
                "count_basis": data.get("count_basis"),
            },
            "models": {
                "trained": models.get("trained_count"),
                "planned": models.get("target_count"),
            },
            "nsga2": {
                "round": nsga.get("round"),
                "candidate_count": nsga.get("candidate_count"),
                "min_volume_L": (nsga.get("summary") or {}).get("min_volume_L"),
            },
            "verification": {
                "stage": verification.get("stage"),
                "valid": (verification.get("counts") or {}).get("valid"),
                "total": (verification.get("counts") or {}).get("total"),
                "final_status": (verification.get("final") or {}).get("status"),
            },
            "scheduler": {
                "connected": scheduler.get("connected"),
                "running": scheduler.get("running"),
                "pending": scheduler.get("pending"),
                "failed": scheduler.get("failed"),
            },
        }

    @staticmethod
    def _snapshot(dashboard: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        """Keep the durable snapshot small; detailed points remain in source artifacts."""
        return {
            **summary,
            "project": dashboard.get("project"),
            "status": dashboard.get("status"),
            "data": {
                **summary["data"],
                "em_valid_rows": dashboard.get("data", {}).get("em_valid_rows"),
                "thermal_valid_rows": dashboard.get("data", {}).get("thermal_valid_rows"),
                "eta_3000": dashboard.get("data", {}).get("eta_3000"),
                "latest_data_at": dashboard.get("data", {}).get("latest_data_at"),
            },
            "models": {
                **summary["models"],
                "items": [
                    {
                        key: model.get(key)
                        for key in ("target", "status", "n_used", "r2", "rmse", "mape_pct", "p90_ape_pct")
                    }
                    for model in dashboard.get("models", {}).get("models", [])
                ],
            },
            "nsga2": {**summary["nsga2"], "summary": dashboard.get("nsga2", {}).get("summary")},
            "verification": {
                **summary["verification"],
                "counts": dashboard.get("verification", {}).get("counts"),
                "agreement": dashboard.get("verification", {}).get("agreement"),
            },
            "scheduler": {**summary["scheduler"], "total": dashboard.get("scheduler", {}).get("total")},
        }

    def record(self, dashboard: dict[str, Any]) -> None:
        summary = self._summary(dashboard)
        signature = (
            summary["overall"],
            summary["data"]["total_rows"],
            summary["data"]["complete_rows"],
            summary["models"]["trained"],
            summary["nsga2"]["round"],
            summary["nsga2"]["candidate_count"],
            summary["verification"]["stage"],
            summary["verification"]["final_status"],
            summary["scheduler"]["running"],
            summary["scheduler"]["pending"],
        )
        now_monotonic = time.monotonic()
        with self._lock:
            if signature == self._last_signature and now_monotonic - self._last_write < self.min_interval_seconds:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            temp = self.snapshot_path.with_suffix(".json.tmp")
            with temp.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(self._snapshot(dashboard, summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temp, self.snapshot_path)
            with self.history_path.open("a", encoding="utf-8", newline="\n") as handle:
                json.dump(summary, handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            self._last_signature = signature
            self._last_write = now_monotonic

    def history(self, limit: int = 2_000) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"entries": [], "warning": None}
        entries: list[dict[str, Any]] = []
        bad_lines = 0
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            entries.append(value)
                    except json.JSONDecodeError:
                        bad_lines += 1
        except OSError as exc:
            return {"entries": [], "warning": f"모니터링 이력 읽기 실패: {exc}"}
        entries = entries[-limit:]
        warning = f"손상된 이력 {bad_lines}줄을 건너뛰었습니다." if bad_lines else None
        return {"entries": entries, "warning": warning}


class ArtifactService:
    """Builds stable JSON view models from live campaign files."""

    def __init__(
        self,
        regression_root: Path,
        scheduler: SchedulerReader | None = None,
        clock: Callable[[], datetime] = _now,
        record_runtime: bool = True,
    ) -> None:
        self.root = Path(regression_root).resolve()
        self.cache = SafeArtifactCache()
        self.scheduler = scheduler or SchedulerReader()
        self.clock = clock
        self.recorder = RuntimeRecorder(self.root / "monitoring" / "runtime") if record_runtime else None

    @staticmethod
    def _warnings(*results: ReadResult) -> list[str]:
        return [result.warning for result in results if result.warning]

    def data(self) -> dict[str, Any]:
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        dataset_dir = self.root / "data" / "dataset"
        manifest_result = self.cache.json(dataset_dir / "manifest.json", {})
        rows_result = self.cache.csv(dataset_dir / "train_io.csv")
        cache_result = self.cache.json(dataset_dir / "collect_cache.json", {})
        strict_result = self.cache.json(
            self.root / "training" / "strict_data_status.json", {}
        )
        manifest = manifest_result.value if isinstance(manifest_result.value, dict) else {}
        rows = rows_result.value if isinstance(rows_result.value, list) else []
        collector_cache = cache_result.value if isinstance(cache_result.value, dict) else {}
        strict_status = (
            strict_result.value if isinstance(strict_result.value, dict) else {}
        )
        warnings = self._warnings(
            manifest_result, rows_result, cache_result, strict_result
        )

        manifest_total = _integer(manifest.get("total_rows"), -1)
        raw_total = manifest_total if manifest_total >= 0 else len(rows)
        if rows and manifest_total >= 0 and len(rows) != manifest_total:
            warnings.append(f"manifest({manifest_total})와 train_io.csv({len(rows)}) 행 수가 다릅니다.")

        identity = strict_status.get("state_identity")
        identity = identity if isinstance(identity, dict) else {}
        pins_valid = all(
            re.fullmatch(r"[0-9a-fA-F]{40}", str(identity.get(key) or ""))
            for key in ("solver_revision", "library_revision")
        )
        strict_available = bool(strict_result.exists and pins_valid)
        total = _integer(strict_status.get("strict_full_rows"), 0) if strict_available else 0
        em_valid = _integer(strict_status.get("strict_em_rows"), 0) if strict_available else 0
        thermal_valid = complete = total
        if not strict_available:
            warnings.append(
                "pinned strict-data status is unavailable; goal progress is fail-closed at zero."
            )

        timestamps = [
            parsed for row in rows
            if (parsed := _parse_time(row.get("saved_at"), now.tzinfo)) is not None
        ]
        timestamps.sort()
        one_hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)
        strict_history = []
        history_path = self.root / "monitoring" / "runtime" / "monitor_history.jsonl"
        if history_path.is_file():
            try:
                with history_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                        except (TypeError, ValueError):
                            continue
                        data_entry = entry.get("data") if isinstance(entry, dict) else None
                        stamp = _parse_time(entry.get("time"), now.tzinfo) if isinstance(entry, dict) else None
                        if (isinstance(data_entry, dict)
                                and data_entry.get("count_basis") == "pinned_strict_full"
                                and stamp is not None):
                            strict_history.append((stamp, _integer(data_entry.get("total_rows"), 0)))
            except OSError as exc:
                warnings.append(f"strict runtime history read failed: {exc}")
        def _growth_since(cutoff):
            earlier = [value for stamp, value in strict_history if stamp <= cutoff]
            if earlier:
                return max(0, total - earlier[-1])
            within = [value for stamp, value in strict_history if stamp >= cutoff]
            return max(0, total - within[0]) if within else 0
        throughput_1h = _growth_since(one_hour_ago)
        added_24h = _growth_since(day_ago)
        strict_updated = _parse_time(strict_status.get("time"), now.tzinfo)
        latest_data = strict_updated if strict_available else None
        first_data = strict_history[0][0] if strict_history else None
        stalled_minutes = max(0.0, (now - latest_data).total_seconds() / 60.0) if latest_data else None
        hourly_rate = float(throughput_1h)
        if hourly_rate <= 0 and added_24h:
            hourly_rate = added_24h / 24.0
        remaining = max(0, DATA_GOAL - total)
        eta_hours = remaining / hourly_rate if hourly_rate > 0 else None
        eta = now + timedelta(hours=eta_hours) if eta_hours is not None else None

        history: list[dict[str, Any]] = []
        previous = 0
        for stamp, count in strict_history:
            history.append({
                "time": _iso(stamp), "added": max(0, count - previous),
                "total": count,
            })
            previous = count
        cumulative = total
        if len(history) > 240:
            stride = math.ceil(len(history) / 240)
            history = history[::stride]
            if history[-1]["total"] != cumulative:
                history.append({
                    "time": _iso(now),
                    "added": max(0, cumulative - history[-1]["total"]),
                    "total": cumulative,
                })

        revisions = Counter(
            str(row.get("git_hash", "")).strip().lower()
            for row in rows if str(row.get("git_hash", "")).strip()
        )
        latest_revision = None
        timed_revisions = [
            (stamp, revision)
            for row in rows
            if (stamp := _parse_time(row.get("saved_at"), now.tzinfo)) is not None
            if (revision := _safe_text(row.get("git_hash"), 40))
        ]
        if timed_revisions:
            latest_revision = max(timed_revisions, key=lambda item: item[0])[1].lower()
        if not latest_revision and revisions:
            latest_revision = revisions.most_common(1)[0][0]
        revision_mismatch = (
            sum(count for revision, count in revisions.items() if revision != latest_revision)
            if latest_revision else 0
        )
        hashes = manifest.get("git_hashes") if isinstance(manifest.get("git_hashes"), list) else []
        harvested = collector_cache.get("harvested")
        nodata = collector_cache.get("nodata")
        local_parts = collector_cache.get("local_parts")

        return {
            "schema_version": SCHEMA_VERSION,
            "available": manifest_result.exists or rows_result.exists,
            "count_basis": "pinned_strict_full",
            "strict_status_available": strict_available,
            "raw_total_rows": raw_total,
            "total_rows": total,
            "em_valid_rows": em_valid,
            "thermal_valid_rows": thermal_valid,
            "complete_rows": complete,
            "em_only_rows": max(0, em_valid - complete),
            "invalid_em_rows": max(0, raw_total - em_valid),
            "manifest_new_rows": _integer(manifest.get("new_rows"), 0),
            "manifest_new_unique_rows": _integer(manifest.get("new_unique_rows"), 0),
            "goal": DATA_GOAL,
            "stretch_goal": STRETCH_GOAL,
            "goal_progress_pct": min(100.0, total / DATA_GOAL * 100.0),
            "stretch_progress_pct": min(100.0, total / STRETCH_GOAL * 100.0),
            "remaining_to_goal": remaining,
            "throughput_1h": throughput_1h,
            "added_24h": added_24h,
            "effective_hourly_rate": hourly_rate,
            "eta_3000": _iso(eta),
            "eta_hours": eta_hours,
            "first_data_at": _iso(first_data),
            "latest_data_at": _iso(latest_data),
            "stalled_minutes": stalled_minutes,
            "stalled": bool(stalled_minutes is not None and stalled_minutes >= 90 and total < DATA_GOAL),
            "latest_revision": latest_revision,
            "revision_count": len(revisions) or len(hashes),
            "rows_not_latest_revision": revision_mismatch,
            "collector": {
                "harvested_tasks": len(harvested) if isinstance(harvested, list) else None,
                "no_data_tasks": len(nodata) if isinstance(nodata, list) else None,
                "local_parts": len(local_parts) if isinstance(local_parts, list) else None,
            },
            "history": history,
            "source": {
                "manifest": str(manifest_result.path),
                "rows": str(rows_result.path),
                "strict_status": str(strict_result.path),
                "updated_at": _iso(
                    strict_result.mtime or manifest_result.mtime or rows_result.mtime
                ),
            },
            "warnings": warnings,
        }

    def models(self, current_data_count: int | None = None) -> dict[str, Any]:
        registry = self.root / "training" / "registry"
        report_result = self.cache.json(registry / "train_report.json", {})
        curve_result = self.cache.csv(self.root / "training" / "learning_curve.csv", max_rows=200_000)
        report_payload = report_result.value if isinstance(report_result.value, dict) else {}
        report = report_payload.get("report") if isinstance(report_payload.get("report"), dict) else {}
        curve_rows = curve_result.value if isinstance(curve_result.value, list) else []
        warnings = self._warnings(report_result, curve_result)
        if current_data_count is None:
            current_data_count = self.data()["total_rows"]

        histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in curve_rows:
            target = _safe_text(row.get("target"), 120)
            if not target or str(row.get("slice", "global")).strip().lower() != "global":
                continue
            item = {
                "time": _safe_text(row.get("time"), 80),
                "n": _integer(row.get("n"), 0),
                "r2": _finite_number(row.get("r2")),
                "rmse": _finite_number(row.get("rmse")),
                "mape_pct": _finite_number(row.get("mape_pct")),
                "p90_ape_pct": _finite_number(row.get("p90_ape_pct")),
            }
            histories[target].append(item)

        ordered_targets = [item["name"] for item in TARGETS]
        ordered_targets.extend(sorted(set(report) - set(ordered_targets)))
        models: list[dict[str, Any]] = []
        latest_times: list[datetime] = []
        for target in ordered_targets:
            metrics = report.get(target) if isinstance(report.get(target), dict) else None
            meta_result = self.cache.json(registry / target / "meta.json", {})
            if meta_result.warning:
                warnings.append(meta_result.warning)
            meta = meta_result.value if isinstance(meta_result.value, dict) else {}
            if metrics is None and isinstance(meta.get("metrics"), dict):
                metrics = meta["metrics"]
            metrics = metrics or {}
            trained = bool(metrics)
            trained_at = _safe_text(meta.get("trained_at") or report_payload.get("time"), 80)
            parsed_trained_at = _parse_time(trained_at, self.clock().tzinfo)
            if parsed_trained_at:
                latest_times.append(parsed_trained_at)
            n_train = _integer(metrics.get("n_train"), 0)
            n_holdout = _integer(metrics.get("n_holdout"), 0)
            n_used = n_train + n_holdout
            stale = bool(trained and current_data_count and n_used and current_data_count >= n_used + max(100, int(n_used * 0.25)))
            history = histories.get(target, [])[-100:]
            previous = history[-1] if history else None
            r2 = _finite_number(metrics.get("r2"))
            mape = _finite_number(metrics.get("mape_pct"))
            attention = bool(trained and ((r2 is not None and r2 < 0.8) or (mape is not None and mape > 20.0)))
            if not trained:
                status = "not_trained"
            elif stale:
                status = "stale"
            elif attention:
                status = "attention"
            else:
                status = "trained"
            models.append({
                "target": target,
                "label": TARGET_META.get(target, {}).get("label", target),
                "unit": TARGET_META.get(target, {}).get("unit", ""),
                "status": status,
                "trained": trained,
                "stale": stale,
                "n_train": n_train,
                "n_holdout": n_holdout,
                "n_used": n_used,
                "r2": r2,
                "rmse": _finite_number(metrics.get("rmse")),
                "mape_pct": mape,
                "p90_ape_pct": _finite_number(metrics.get("p90_ape_pct")),
                "q90_conformal": _finite_number(metrics.get("q90_conformal") or meta.get("q90")),
                "trained_at": trained_at,
                "delta_r2": (r2 - previous["r2"] if r2 is not None and previous and previous["r2"] is not None else None),
                "delta_mape_pct": (mape - previous["mape_pct"] if mape is not None and previous and previous["mape_pct"] is not None else None),
                "history": history,
            })

        trained_count = sum(model["trained"] for model in models)
        return {
            "schema_version": SCHEMA_VERSION,
            "available": report_result.exists or curve_result.exists,
            "target_count": len(models),
            "trained_count": trained_count,
            "missing_count": len(models) - trained_count,
            "current_data_count": current_data_count,
            "latest_trained_at": _iso(max(latest_times)) if latest_times else _safe_text(report_payload.get("time"), 80),
            "quality_note": "주의 표시는 탐색용 기준(R² < 0.8 또는 MAPE > 20%)이며, 최종 합격은 독립 FEA 검증으로 판정합니다.",
            "models": models,
            "warnings": list(dict.fromkeys(warnings)),
            "source": str(report_result.path),
        }

    def model_history(self, target: str) -> dict[str, Any] | None:
        if target not in TARGET_META:
            return None
        models = self.models()
        model = next((item for item in models["models"] if item["target"] == target), None)
        return {"target": target, "label": TARGET_META[target]["label"], "history": model["history"] if model else []}

    @staticmethod
    def _constraint(value: float | None, limit: float | tuple[float, float], mode: str) -> dict[str, Any]:
        if value is None:
            return {"value": None, "limit": limit, "margin": None, "pass": None}
        if mode == "max":
            limit_value = float(limit)
            return {"value": value, "limit": limit_value, "margin": limit_value - value, "pass": value <= limit_value}
        if mode == "min":
            limit_value = float(limit)
            return {"value": value, "limit": limit_value, "margin": value - limit_value, "pass": value >= limit_value}
        low, high = limit
        return {
            "value": value,
            "limit": [low, high],
            "margin": min(value - low, high - value),
            "pass": low <= value <= high,
        }

    def _candidate(self, row: dict[str, Any], round_number: int, index: int) -> dict[str, Any]:
        volume = _finite_number(row.get("volume_L"))
        loss = _finite_number(row.get("total_loss_W"))
        llt = _finite_number(row.get("pred_Llt_phys"))
        bmax = _finite_number(row.get("pred_B_max_core"))
        temperatures = {
            target: _finite_number(row.get(f"pred_{target}")) for target in TEMPERATURE_TARGETS
        }
        available_temperatures = [value for value in temperatures.values() if value is not None]
        max_temperature = max(available_temperatures) if available_temperatures else None
        insulation_values = [
            value for key in INSULATION_KEYS if (value := _finite_number(row.get(key))) is not None
        ]
        min_insulation = min(insulation_values) if insulation_values else None
        constraints = {
            "llt": self._constraint(llt, (26.95, 28.05), "band"),
            "temperature": self._constraint(max_temperature, 100.0, "max"),
            "bmax": self._constraint(bmax, 1.2, "max"),
            "insulation": self._constraint(min_insulation, 40.0, "min"),
        }
        passes = [item["pass"] for item in constraints.values()]
        spec_status = "fail" if False in passes else ("pass" if passes and all(value is True for value in passes) else "unknown")
        sigmas = {
            key.removeprefix("sigma_"): value
            for key, raw in row.items()
            if key.startswith("sigma_") and (value := _finite_number(raw)) is not None
        }
        parameters = {
            key: _coerce(row.get(key)) for key in DESIGN_PARAMETER_KEYS if row.get(key) not in (None, "")
        }
        return {
            "id": f"r{round_number:02d}-{index:04d}",
            "index": index,
            "round": round_number,
            "volume_L": volume,
            "total_loss_W": loss,
            "pred_Llt_phys": llt,
            "pred_B_max_core": bmax,
            "pred_max_temperature_C": max_temperature,
            "pred_temperatures_C": temperatures,
            "min_insulation_mm": min_insulation,
            "constraints": constraints,
            "spec_status": spec_status,
            "uncertainty": sigmas,
            "parameters": parameters,
        }

    def _round_directories(self) -> list[tuple[int, Path]]:
        base = self.root / "al_rounds"
        if not base.is_dir():
            return []
        found = []
        for child in base.iterdir():
            match = re.fullmatch(r"round_(\d+)", child.name)
            if match and child.is_dir() and (child / "pareto_front.csv").exists():
                found.append((int(match.group(1)), child))
        return sorted(found)

    def nsga2(self) -> dict[str, Any]:
        state_result = self.cache.json(self.root / "al_rounds" / "state.json", {})
        state = state_result.value if isinstance(state_result.value, dict) else {}
        rounds = self._round_directories()
        warnings = self._warnings(state_result)
        if not rounds:
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "waiting",
                "round": _integer(state.get("round"), 0) if state else None,
                "al_stage": _safe_text(state.get("stage"), 30),
                "candidate_count": 0,
                "configured_restarts": 16,
                "completed_restarts": None,
                "candidates": [],
                "summary": {},
                "rounds": [],
                "warnings": warnings,
            }

        round_summaries: list[dict[str, Any]] = []
        parsed_rounds: dict[int, tuple[ReadResult, list[dict[str, str]]]] = {}
        for number, directory in rounds:
            result = self.cache.csv(directory / "pareto_front.csv", max_rows=20_000)
            rows = result.value if isinstance(result.value, list) else []
            parsed_rounds[number] = (result, rows)
            if result.warning:
                warnings.append(result.warning)
            volumes = [value for row in rows if (value := _finite_number(row.get("volume_L"))) is not None]
            losses = [value for row in rows if (value := _finite_number(row.get("total_loss_W"))) is not None]
            round_summaries.append({
                "round": number,
                "candidate_count": len(rows),
                "min_volume_L": min(volumes) if volumes else None,
                "min_loss_W": min(losses) if losses else None,
                "updated_at": _iso(result.mtime),
            })

        latest_round, latest_dir = rounds[-1]
        latest_result, latest_rows = parsed_rounds[latest_round]
        candidates = [self._candidate(row, latest_round, index) for index, row in enumerate(latest_rows)]
        volume_candidates = [item for item in candidates if item["volume_L"] is not None]
        loss_candidates = [item for item in candidates if item["total_loss_W"] is not None]
        minimum_volume = min(volume_candidates, key=lambda item: item["volume_L"]) if volume_candidates else None
        minimum_loss = min(loss_candidates, key=lambda item: item["total_loss_W"]) if loss_candidates else None
        for candidate in candidates:
            candidate["is_min_volume"] = bool(minimum_volume and candidate["id"] == minimum_volume["id"])
            candidate["is_min_loss"] = bool(minimum_loss and candidate["id"] == minimum_loss["id"])
        candidates.sort(key=lambda item: (item["volume_L"] is None, item["volume_L"] or 0.0))

        latest_summary = round_summaries[-1]
        previous_summary = round_summaries[-2] if len(round_summaries) > 1 else None
        comparison = None
        if previous_summary:
            comparison = {
                "previous_round": previous_summary["round"],
                "min_volume_change_L": (
                    latest_summary["min_volume_L"] - previous_summary["min_volume_L"]
                    if latest_summary["min_volume_L"] is not None and previous_summary["min_volume_L"] is not None else None
                ),
                "min_loss_change_W": (
                    latest_summary["min_loss_W"] - previous_summary["min_loss_W"]
                    if latest_summary["min_loss_W"] is not None and previous_summary["min_loss_W"] is not None else None
                ),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "available": True,
            "status": "running" if str(state.get("stage", "")).upper() == "OPTIMIZE" else "completed",
            "round": latest_round,
            "al_round": _integer(state.get("round"), latest_round) if state else latest_round,
            "al_stage": _safe_text(state.get("stage"), 30),
            "candidate_count": len(candidates),
            "configured_restarts": 16,
            "completed_restarts": state.get("nsga2_restarts_completed") if state else None,
            "candidates": candidates,
            "summary": {
                **latest_summary,
                "min_volume_candidate_id": minimum_volume["id"] if minimum_volume else None,
                "min_loss_candidate_id": minimum_loss["id"] if minimum_loss else None,
                "known_spec_pass_count": sum(item["spec_status"] == "pass" for item in candidates),
                "known_spec_fail_count": sum(item["spec_status"] == "fail" for item in candidates),
                "unknown_spec_count": sum(item["spec_status"] == "unknown" for item in candidates),
            },
            "comparison": comparison,
            "rounds": round_summaries,
            "source": str(latest_dir / "pareto_front.csv"),
            "updated_at": _iso(latest_result.mtime),
            "note": "Pareto 파일의 해는 최적화기가 feasible로 반환한 후보입니다. 아직 학습되지 않은 출력의 명목 사양 판정은 ‘확인 불가’로 표시합니다.",
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _evaluate_fea(self, result: dict[str, Any], require_full_model: bool = False) -> dict[str, Any]:
        full_model_value = _finite_number(result.get("full_model"))
        llt = _finite_number(result.get("Llt_phys"))
        if llt is None:
            raw_llt = _finite_number(result.get("Llt"))
            if raw_llt is not None:
                llt = raw_llt * (1.0 if full_model_value == 1.0 else 2.0)
        bmax = _finite_number(result.get("B_max_core"))
        n2_side = _finite_number(result.get("N2_side"))
        temperature_keys = ["T_max_Tx", "T_max_Rx_main", "T_max_core"]
        if n2_side is not None and n2_side > 0:
            temperature_keys.insert(2, "T_max_Rx_side")
        temperatures = {key: _finite_number(result.get(key)) for key in temperature_keys}
        finite_temperatures = [value for value in temperatures.values() if value is not None]
        max_temperature = max(finite_temperatures) if len(finite_temperatures) == len(temperature_keys) else None
        insulation_values = [
            value for key in INSULATION_KEYS if (value := _finite_number(result.get(key))) is not None
        ]
        min_insulation = min(insulation_values) if insulation_values else None
        matrix_error = _finite_number(result.get("conv_error_pct_matrix"))
        loss_error = _finite_number(result.get("conv_error_pct_loss"))
        convergence_value = max(matrix_error, loss_error) if matrix_error is not None and loss_error is not None else None
        checks = {
            "llt": self._constraint(llt, (26.95, 28.05), "band"),
            "temperature": self._constraint(max_temperature, 100.0, "max"),
            "bmax": self._constraint(bmax, 1.2, "max"),
            "insulation": self._constraint(min_insulation, 40.0, "min"),
            "convergence": self._constraint(convergence_value, 1.5, "max"),
        }
        if require_full_model:
            checks["full_model"] = {
                "value": full_model_value,
                "limit": 1,
                "margin": None,
                "pass": full_model_value == 1.0 if full_model_value is not None else None,
            }
        states = [item["pass"] for item in checks.values()]
        computed_status = "fail" if False in states else ("pass" if states and all(value is True for value in states) else "unknown")
        loss_components = [
            _finite_number(result.get(key)) for key in
            ("P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total")
        ]
        total_loss = sum(value for value in loss_components if value is not None) if all(value is not None for value in loss_components) else None
        return {
            "computed_status": computed_status,
            "checks": checks,
            "Llt_phys_uH": llt,
            "B_max_core_T": bmax,
            "max_temperature_C": max_temperature,
            "temperatures_C": temperatures,
            "min_insulation_mm": min_insulation,
            "total_loss_W": total_loss,
            "volume_L": _finite_number(result.get("volume_L")),
            "conv_error_pct_matrix": matrix_error,
            "conv_error_pct_loss": loss_error,
            "solver_revision": _safe_text(result.get("git_hash"), 40),
            "library_revision": _safe_text(result.get("pyaedt_library_git_hash"), 40),
            "parameters": {
                key: _coerce(result.get(key)) for key in DESIGN_PARAMETER_KEYS if result.get(key) not in (None, "")
            },
        }

    def _final_artifact(self) -> tuple[ReadResult, dict[str, Any]]:
        paths = (
            self.root / "verify" / "results" / "final_verification.json",
            self.root / "verify" / "final_verification.json",
            self.root / "monitoring" / "runtime" / "final_verification.json",
        )
        for path in paths:
            if path.exists():
                result = self.cache.json(path, {})
                value = result.value if isinstance(result.value, dict) else {}
                return result, value
        return ReadResult({}, str(paths[0]), False), {}

    def verification(self, nsga: dict[str, Any] | None = None) -> dict[str, Any]:
        nsga = nsga or self.nsga2()
        state_result = self.cache.json(self.root / "al_rounds" / "state.json", {})
        state = state_result.value if isinstance(state_result.value, dict) else {}
        warnings = self._warnings(state_result)
        records = state.get("task_records") if isinstance(state.get("task_records"), dict) else {}
        candidates_by_index = {item["index"]: item for item in nsga.get("candidates", [])}
        standard: list[dict[str, Any]] = []
        for index_text, record_value in records.items():
            if not isinstance(record_value, dict):
                continue
            index = _integer(index_text, -1)
            result = record_value.get("result") if isinstance(record_value.get("result"), dict) else None
            evaluation = self._evaluate_fea(result) if result else None
            predicted = candidates_by_index.get(index)
            standard.append({
                "candidate_id": predicted.get("id") if predicted else f"r{_integer(state.get('round'), 0):02d}-{max(index, 0):04d}",
                "index": index,
                "profile": "standard",
                "task_id": record_value.get("active_id") or record_value.get("original_id"),
                "task_status": _safe_text(record_value.get("last_status"), 30),
                "outcome": _safe_text(record_value.get("outcome"), 50),
                "attempt": _integer(record_value.get("attempt"), 0),
                "evaluation": evaluation,
                "predicted": predicted,
                "error": _safe_text(record_value.get("fetch_error") or record_value.get("error"), 500),
            })

        fine_records = state.get("fine_task_records") \
            if isinstance(state.get("fine_task_records"), dict) else {}
        fine_queue = state.get("final_candidates") \
            if isinstance(state.get("final_candidates"), list) else []
        fine_candidates: list[dict[str, Any]] = []
        for rank_text, record_value in fine_records.items():
            if not isinstance(record_value, dict):
                continue
            rank = _integer(rank_text, -1)
            candidate = fine_queue[rank] if 0 <= rank < len(fine_queue) \
                and isinstance(fine_queue[rank], dict) else {}
            result = record_value.get("result") \
                if isinstance(record_value.get("result"), dict) else None
            fine_candidates.append({
                "rank": rank,
                "candidate_id": _safe_text(candidate.get("candidate_digest"), 100),
                "volume_L": _finite_number(candidate.get("volume_L")),
                "profile": "fine",
                "task_id": record_value.get("active_id") or record_value.get("original_id"),
                "task_status": _safe_text(record_value.get("last_status"), 30),
                "outcome": _safe_text(record_value.get("outcome"), 50),
                "attempt": _integer(record_value.get("attempt"), 0),
                "evaluation": self._evaluate_fea(result, require_full_model=True) if result else None,
                "error": _safe_text(
                    record_value.get("unverified_reason")
                    or record_value.get("fetch_error")
                    or record_value.get("error"), 500,
                ),
            })

        verification_counts = state.get("verification_counts") if isinstance(state.get("verification_counts"), dict) else {}
        if verification_counts:
            counts = {
                "total": _integer(verification_counts.get("total"), len(records)),
                "valid": _integer(verification_counts.get("valid"), 0),
                "pending": _integer(verification_counts.get("pending"), 0),
                "exhausted": _integer(verification_counts.get("exhausted"), 0),
                "ingested": _integer(verification_counts.get("ingested"), 0),
            }
        else:
            counts = {
                "total": len(records),
                "valid": sum(item.get("outcome") == "valid" for item in records.values() if isinstance(item, dict)),
                "pending": sum(item.get("outcome") in {None, "pending", "fetch_error", "submission_unknown"} for item in records.values() if isinstance(item, dict)),
                "exhausted": sum(item.get("outcome") == "exhausted" for item in records.values() if isinstance(item, dict)),
                "ingested": 0,
            }
        counts["coverage"] = counts["valid"] / counts["total"] if counts["total"] else None

        error_files = sorted((self.root / "al_rounds").glob("round_*/verification_errors.csv"))
        errors: list[dict[str, Any]] = []
        error_source = None
        if error_files:
            error_result = self.cache.csv(error_files[-1], max_rows=10_000)
            error_source = str(error_files[-1])
            errors = [
                {key: _coerce(value) for key, value in row.items()}
                for row in (error_result.value if isinstance(error_result.value, list) else [])
            ]
            if error_result.warning:
                warnings.append(error_result.warning)

        final_result, final_payload = self._final_artifact()
        if final_result.warning:
            warnings.append(final_result.warning)
        raw_final_result = final_payload.get("result") if isinstance(final_payload.get("result"), dict) else final_payload
        final_evaluation = self._evaluate_fea(raw_final_result, require_full_model=True) if final_payload else None
        declared = _safe_text(final_payload.get("status"), 30) if final_payload else None
        declared_pass = final_payload.get("passed", final_payload.get("overall_pass")) if final_payload else None
        declared_success = declared_pass is True or (
            declared and declared.lower() in {"pass", "passed", "complete", "completed"}
        )
        declared_failure = declared_pass is False or (
            declared and declared.lower() in {"fail", "failed", "error"}
        )
        # Never let a manually declared PASS override a physical check.  A
        # partial result remains unknown until every required value is present.
        if declared_failure or (final_evaluation and final_evaluation["computed_status"] == "fail"):
            final_status = "fail"
        elif final_evaluation and final_evaluation["computed_status"] == "pass":
            final_status = "pass"
        elif declared_success or final_payload:
            final_status = "unknown"
        elif state.get("stage") == "FINE_BLOCKED":
            final_status = "blocked"
        else:
            final_status = "waiting"
        final = {
            "available": bool(final_payload),
            "status": final_status,
            "candidate_id": _safe_text(final_payload.get("candidate_id"), 100) if final_payload else None,
            "profile": _safe_text(final_payload.get("profile"), 30) or ("fine" if final_payload else None),
            "task_id": (
                final_payload.get("fine_task_id") or final_payload.get("task_id")
            ) if final_payload else None,
            "task_status": _safe_text(
                final_payload.get("fine_task_status") or final_payload.get("task_status"), 30
            ) if final_payload else None,
            "evaluation": final_evaluation,
            "declared_status": declared,
            "error": _safe_text(
                (final_payload.get("error") or final_payload.get("failure_reason"))
                if final_payload else state.get("fine_block_reason"), 1000,
            ),
            "updated_at": _safe_text(
                final_payload.get("generated_at") or final_payload.get("updated_at")
                or final_payload.get("time"), 80,
            ) if final_payload else None,
            "source": final_result.path if final_result.exists else None,
        }

        history = state.get("history") if isinstance(state.get("history"), list) else []
        agreement = history[-1] if history and isinstance(history[-1], dict) else None
        return {
            "schema_version": SCHEMA_VERSION,
            "available": bool(state or standard or errors or final_payload),
            "stage": _safe_text(state.get("stage"), 30) or "NOT_STARTED",
            "round": _integer(state.get("round"), 0) if state else None,
            "counts": counts,
            "standard_candidates": standard,
            "fine_candidates": fine_candidates,
            "verification_errors": errors,
            "agreement": agreement,
            "final": final,
            "sources": {"state": state_result.path if state_result.exists else None, "errors": error_source},
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _status(
        self,
        data: dict[str, Any],
        models: dict[str, Any],
        nsga: dict[str, Any],
        verification: dict[str, Any],
        scheduler: dict[str, Any],
    ) -> dict[str, Any]:
        stages = []
        simulation_active = scheduler.get("running", 0) + scheduler.get("pending", 0) > 0
        if simulation_active:
            simulation_state = "active"
            simulation_detail = f"실행 {scheduler.get('running', 0)} · 대기 {scheduler.get('pending', 0)}"
        elif not scheduler.get("connected"):
            simulation_state = "warning"
            simulation_detail = "스케줄러 상태 확인 불가"
        else:
            simulation_state = "waiting"
            simulation_detail = "실행 중 작업 없음"
        stages.append({"key": "simulation", "label": "시뮬레이션", "state": simulation_state, "detail": simulation_detail})

        if data["total_rows"] >= DATA_GOAL:
            data_state, data_detail = "complete", f"목표 달성 · {data['total_rows']:,}개"
        elif data["throughput_1h"] > 0:
            data_state, data_detail = "active", f"최근 1시간 +{data['throughput_1h']}개"
        elif data["stalled"]:
            data_state, data_detail = "warning", "90분 이상 데이터 증가 없음"
        else:
            data_state, data_detail = "waiting", f"{data['total_rows']:,}개 확보"
        stages.append({"key": "data", "label": "데이터 적재", "state": data_state, "detail": data_detail})

        if models["trained_count"] == 0:
            model_state, model_detail = "waiting", "학습 모델 없음"
        elif models["missing_count"]:
            model_state, model_detail = "warning", f"{models['trained_count']}/{models['target_count']} 모델 학습"
        else:
            model_state, model_detail = "complete", f"{models['trained_count']}개 모델 준비"
        stages.append({"key": "models", "label": "모델 학습", "state": model_state, "detail": model_detail})

        nsga_state = "active" if nsga["status"] == "running" else ("complete" if nsga["available"] else "waiting")
        nsga_detail = f"round {nsga.get('round')} · {nsga['candidate_count']}개" if nsga["available"] else "실행 전"
        stages.append({"key": "nsga2", "label": "NSGA-II", "state": nsga_state, "detail": nsga_detail})

        verification_stage = str(verification.get("stage", "NOT_STARTED")).upper()
        if verification["counts"]["total"]:
            verify_state = "active" if verification["counts"]["pending"] else "complete"
            verify_detail = f"유효 {verification['counts']['valid']}/{verification['counts']['total']}"
        elif verification_stage in {"SUBMIT", "WAIT", "INGEST", "CHECK"}:
            verify_state, verify_detail = "active", verification_stage
        else:
            verify_state, verify_detail = "waiting", "검증 전"
        stages.append({"key": "verification", "label": "후보 FEA", "state": verify_state, "detail": verify_detail})

        final_status = verification["final"]["status"]
        final_state = {"pass": "complete", "fail": "error"}.get(final_status, "waiting")
        final_detail = {"pass": "최종 설계 확정", "fail": "fine FEA 실패"}.get(final_status, "검증 전")
        stages.append({"key": "final", "label": "최종 설계", "state": final_state, "detail": final_detail})

        warnings: list[str] = []
        for payload in (data, models, nsga, verification):
            warnings.extend(payload.get("warnings", []))
        if not scheduler.get("connected") and scheduler.get("error"):
            warnings.append(scheduler["error"])
        if data["stalled"]:
            warnings.append(f"유효 데이터가 약 {data['stalled_minutes']:.0f}분 동안 증가하지 않았습니다.")
        if models["missing_count"]:
            missing = [model["label"] for model in models["models"] if not model["trained"]]
            warnings.append("미학습 모델: " + ", ".join(missing))
        if final_status == "fail":
            warnings.append("최종 fine FEA가 사양을 통과하지 못했습니다.")
        warnings = list(dict.fromkeys(warnings))

        if final_status == "fail" or any(stage["state"] == "error" for stage in stages):
            overall = "error"
        elif warnings:
            overall = "warning"
        elif any(stage["state"] == "active" for stage in stages):
            overall = "active"
        else:
            overall = "idle"
        current = next((stage for stage in reversed(stages) if stage["state"] == "active"), None)
        if current is None:
            current = next((stage for stage in stages if stage["state"] in {"warning", "error"}), stages[0])
        return {
            "overall": overall,
            "current_stage": current["key"],
            "current_stage_label": current["label"],
            "stages": stages,
            "warnings": warnings,
        }

    def dashboard(self, record: bool = True) -> dict[str, Any]:
        generated_at = _iso(self.clock())
        data = self.data()
        models = self.models(data["total_rows"])
        nsga = self.nsga2()
        verification = self.verification(nsga)
        scheduler = self.scheduler.snapshot()
        dashboard = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "project": "MFT_1MW_2026",
            "status": self._status(data, models, nsga, verification, scheduler),
            "data": data,
            "models": models,
            "nsga2": nsga,
            "verification": verification,
            "scheduler": scheduler,
        }
        if record and self.recorder:
            try:
                self.recorder.record(dashboard)
            except (OSError, TypeError, ValueError) as exc:
                dashboard["status"]["warnings"].append(f"모니터 이력 기록 실패: {exc}")
                if dashboard["status"]["overall"] not in {"error"}:
                    dashboard["status"]["overall"] = "warning"
        return dashboard

    def status(self) -> dict[str, Any]:
        dashboard = self.dashboard(record=False)
        return {
            "schema_version": dashboard["schema_version"],
            "generated_at": dashboard["generated_at"],
            "project": dashboard["project"],
            **dashboard["status"],
            "scheduler": dashboard["scheduler"],
        }

    def history(self) -> dict[str, Any]:
        if not self.recorder:
            return {"entries": [], "warning": None}
        return self.recorder.history()
