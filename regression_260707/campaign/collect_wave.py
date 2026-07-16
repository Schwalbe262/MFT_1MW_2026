"""
캠페인 결과 회수·병합기.

완료된 mft-camp-* 태스크의 stdout에서 ===RESULT_CSV=== 블록을 파싱해
스키마-유니온으로 병합하고, 수렴 필터·중복 제거 후 dataset/train.parquet에 축적한다.

사용:
  python collect_wave.py --prefix mft-camp-w1          # 웨이브 1 회수
  python collect_wave.py --prefix mft-camp --all       # 전체 회수
"""
import argparse
import glob
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime

import pandas as pd
import requests
from filelock import FileLock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from module.thermal_probe_contract import (
    RX_SIDE_FACE_MAX_RULE,
    RX_SIDE_FACE_MEAN_RULE,
    RX_SIDE_FACE_PROBE_CONTRACT_VERSION,
)
from module.core_material_contract import PHYSICS_DATA_REVISION

if __package__:
    from .train_io import build_train_io
    try:
        from ..model_targets import CORE_REGION_TEMPERATURE_TARGETS
    except ImportError:  # ``campaign`` imported with regression root on sys.path.
        from model_targets import CORE_REGION_TEMPERATURE_TARGETS
else:  # Direct execution: python campaign/collect_wave.py
    from train_io import build_train_io
    regression_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    if regression_root not in sys.path:
        sys.path.insert(0, regression_root)
    from model_targets import CORE_REGION_TEMPERATURE_TARGETS

DEFAULT_SCHEDULER = "http://127.0.0.1:8000"
LOCAL_SCHEDULER_FALLBACK = "http://127.0.0.1:8001"


def _configured_scheduler_url():
    """Resolve the scheduler endpoint once for this collector subprocess."""

    return (
        os.environ.get("MFT_SCHEDULER_URL", DEFAULT_SCHEDULER).strip().rstrip("/")
        or DEFAULT_SCHEDULER
    )


SCHEDULER = _configured_scheduler_url()
TASK_LIST_LIMIT = 2000
TASK_LIST_MAX_PAGES = 10000
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "data", "dataset")
LOCAL_RESULTS_CSV = os.path.join(HERE, "..", "..", "simulation_results_260706.csv")
LOCAL_RESULTS_PARTS_DIR = os.path.join(HERE, "..", "..", "results_parts_260706")
FEEDER_STATE_PATH = os.path.join(HERE, "feeder_state.json")

SOURCE_RANK_COLUMN = "_collector_source_rank"
SOURCE_RANK_TERMINAL_CSV = 10
SOURCE_RANK_JSON = 20
SOURCE_RANK_LOCAL_CSV = 30
SOURCE_RANK_LOCAL_PART = 40
# A manually requested policy replay must be able to replace the same raw
# scheduler payload after a collector bug is fixed. The one-point bump keeps
# normal source precedence intact while making a second replay a no-op.
SOURCE_RANK_RECOLLECT_OFFSET = 1

FETCH_ATTEMPTS = 3
RETRY_BASE_SECONDS = 0.5
RETRY_MAX_SECONDS = 30.0
MAX_STDOUT_BYTES = 1_048_576
MAX_TRUSTED_TEMPERATURE_C = 4700.0
MIN_TRUSTED_TEMPERATURE_C = -273.15
MANDATORY_THERMAL_TEMPERATURE_COLUMNS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_core_center_max",
    *CORE_REGION_TEMPERATURE_TARGETS,
)
SIDE_THERMAL_TEMPERATURE_COLUMNS = (
    "T_max_Rx_side",
    "Tprobe_Rx_side_leeward_max",
)
STACKING_REVISION_SIDE_THERMAL_TEMPERATURE_COLUMNS = (
    "Tprobe_Rx_side_leeward_mean",
    "Tprobe_Rx_side_outer_max",
    "Tprobe_Rx_side_outer_mean",
    "Tprobe_Rx_side_inner_max",
    "Tprobe_Rx_side_inner_mean",
)


class FetchError(RuntimeError):
    """A scheduler response could not be fetched reliably."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _retry_delay(response, attempt):
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(RETRY_MAX_SECONDS, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** attempt))


def _get_response(path, *, params=None, timeout=30, attempts=FETCH_ATTEMPTS):
    """GET with bounded retry for 429, 5xx, and connection failures."""
    global SCHEDULER
    last_error = None
    for attempt in range(attempts):
        response = None
        if path.startswith("http"):
            request_targets = [(None, path)]
        else:
            request_targets = [(SCHEDULER, f"{SCHEDULER}{path}")]
            # The local scheduler may be moved to the recovery listener while
            # a recurring collector is still configured for the legacy port.
            # Never redirect an explicit remote endpoint to localhost.
            if SCHEDULER == DEFAULT_SCHEDULER:
                request_targets.append((
                    LOCAL_SCHEDULER_FALLBACK,
                    f"{LOCAL_SCHEDULER_FALLBACK}{path}",
                ))
        for scheduler_base, url in request_targets:
            try:
                response = requests.get(url, params=params, timeout=timeout)
            except requests.RequestException as exc:
                last_error = FetchError(f"GET {path} failed: {exc}")
                continue
            if scheduler_base is not None and scheduler_base != SCHEDULER:
                SCHEDULER = scheduler_base
            break
        if response is not None:
            status_code = int(response.status_code)
            if status_code != 429 and status_code < 500:
                if status_code >= 400:
                    raise FetchError(
                        f"GET {path} returned HTTP {status_code}", status_code
                    )
                return response
            last_error = FetchError(
                f"GET {path} returned HTTP {status_code}", status_code
            )
        if attempt + 1 < attempts:
            time.sleep(_retry_delay(response, attempt))
    raise last_error or FetchError(f"GET {path} failed")


def _get_json(path, *, params=None, timeout=30):
    response = _get_response(path, params=params, timeout=timeout)
    try:
        return response.json()
    except (TypeError, ValueError) as exc:
        raise FetchError(f"GET {path} returned invalid JSON: {exc}") from exc


def fetch_stdout(task_id, timeout=30):
    return _get_response(
        f"/api/tasks/{task_id}/stdout",
        params={"max_bytes": MAX_STDOUT_BYTES}, timeout=timeout).text


def _legacy_list_tasks_with_detail_probes(prefix):
    # 신 스케줄러(limit/name_prefix 지원) 우선
    # A 10k response takes longer than the 30 s client timeout on the live
    # scheduler.  The durable feeder ledger below recovers every task outside
    # this bounded recent window, so a smaller page reduces latency without
    # dropping outstanding campaign work.
    t = _get_json("/api/tasks",
                  params={"limit": TASK_LIST_LIMIT,
                          "name_prefix": prefix}, timeout=30)
    tasks = t if isinstance(t, list) else t.get("tasks", [])
    matched = [x for x in tasks if str(x.get("name", "")).startswith(prefix)]
    page_seen = {x["id"]: x for x in matched}
    seen = dict(page_seen)

    def probe(tid):
        try:
            x = _get_json(f"/api/tasks/{tid}", timeout=15)
        except FetchError as exc:
            return False if exc.status_code == 404 else None
        return x if str(x.get("name", "")).startswith(prefix) else False

    # The scheduler has no pagination and applies its limit before prefix
    # filtering. Always recover unseen feeder IDs from the durable local ledger.
    try:
        with open(FEEDER_STATE_PATH, encoding="utf-8") as stream:
            feeder_state = json.load(stream)
        ledger_ids = {int(task_id) for task_id in feeder_state.get("outstanding", [])}
    except FileNotFoundError:
        ledger_ids = set()
    except (OSError, ValueError, TypeError) as exc:
        raise FetchError(f"feeder task ledger is unreadable: {exc}") from exc
    cache = _load_cache()
    judged = set(cache.get("nodata", [])) | set(cache.get("harvested", []))
    missing_ids = sorted(ledger_ids - set(seen) - judged)[:500]
    for task_id in missing_ids:
        recovered = probe(task_id)
        if recovered:
            seen[task_id] = recovered
    if len(tasks) > 250:
        return list(seen.values())

    # 구 스케줄러(200개 페이지): ID 연속 스캔으로 누락 보완
    if not page_seen:
        return list(seen.values())

    scan_seen = dict(page_seen)
    ids = sorted(scan_seen)
    lo, hi = ids[0], ids[-1]
    # 경계 확장: 프리픽스 연속 구간 가정, 밖으로 miss 20회까지
    for direction in (-1, +1):
        cur = lo if direction < 0 else hi
        misses = 0
        while misses < 20 and cur > 0:
            cur += direction
            r = probe(cur)
            if r:
                seen[cur] = r
                scan_seen[cur] = r
                misses = 0
            elif r is False:
                misses += 1
            else:
                break
    # 범위 내 구멍 채우기 (페이지에 안 담긴 중간 ID) - 이미 판정된 터미널 ID는 생략
    try:
        c = _load_cache()
        judged = set(c.get("nodata", [])) | set(c.get("harvested", []))
    except Exception:
        judged = set()
    lo, hi = min(scan_seen), max(scan_seen)
    for tid in range(lo, hi + 1):
        if tid not in seen and tid not in judged:
            r = probe(tid)
            if r:
                seen[tid] = r
    return list(seen.values())


def list_tasks(prefix):
    """Return every matching scheduler task without per-task detail probes.

    The scheduler's compact inventory applies ``name_prefix`` before its
    limit and exposes a descending ``before_id`` cursor. Walking that cursor
    is complete and bounded. The former ID-by-ID recovery path generated
    hundreds of requests for small campaign prefixes and could starve the web
    UI and AEDT heartbeat handlers.
    """

    prefix = str(prefix or "")
    if not prefix:
        raise ValueError("task prefix must be non-empty")

    seen = {}
    before_id = 0
    for _page_number in range(TASK_LIST_MAX_PAGES):
        params = {
            "compact": "true",
            "limit": TASK_LIST_LIMIT,
            "name_prefix": prefix,
        }
        if before_id:
            params["before_id"] = before_id
        payload = _get_json("/api/tasks", params=params, timeout=30)
        tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
        if not isinstance(tasks, list):
            raise FetchError("GET /api/tasks returned a non-list task inventory")

        page_ids = []
        for task in tasks:
            if not isinstance(task, dict):
                raise FetchError("GET /api/tasks returned a non-object task row")
            try:
                task_id = int(task["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FetchError("GET /api/tasks returned an invalid task ID") from exc
            if task_id <= 0:
                raise FetchError("GET /api/tasks returned a non-positive task ID")
            page_ids.append(task_id)
            if str(task.get("name", "")).startswith(prefix):
                seen[task_id] = task

        if len(tasks) < TASK_LIST_LIMIT:
            return list(seen.values())
        if not page_ids:
            raise FetchError("GET /api/tasks returned a full page without task IDs")
        next_before_id = min(page_ids)
        if before_id and next_before_id >= before_id:
            raise FetchError("GET /api/tasks cursor did not advance")
        before_id = next_before_id

    raise FetchError("GET /api/tasks exceeded the compact pagination limit")


def list_tasks_for_prefixes(prefixes):
    """Return the union of scheduler tasks for an explicit prefix allowlist."""
    normalized = []
    for value in prefixes:
        prefix = str(value or "").strip()
        if not prefix:
            raise ValueError("task prefixes must be non-empty")
        if prefix not in normalized:
            normalized.append(prefix)

    tasks_by_id = {}
    for prefix in normalized:
        for task in list_tasks(prefix):
            tasks_by_id[int(task["id"])] = task
    return sorted(tasks_by_id.values(), key=lambda task: int(task["id"]))


# 터미널 태스크의 회수 결과 캐시: 재수집 시 재조회 생략 (회수 시간 수분 -> 초)
CACHE_PATH = os.path.join(DATASET_DIR, "collect_cache.json")
CACHE_WRITE_ATTEMPTS = 5
CACHE_RETRY_SECONDS = 0.05


def _empty_cache():
    return {"nodata": [], "harvested": [], "local_parts": []}


def _validated_cache(payload):
    if not isinstance(payload, dict):
        raise ValueError("collector cache root must be an object")
    cache = dict(payload)
    for key in ("nodata", "harvested"):
        values = cache.setdefault(key, [])
        if not isinstance(values, list):
            raise ValueError(f"collector cache {key} must be a list")
        for task_id in values:
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
                raise ValueError(f"collector cache {key} contains an invalid task ID")
    local_parts = cache.setdefault("local_parts", [])
    if (not isinstance(local_parts, list)
            or any(not isinstance(part, str) or not part for part in local_parts)):
        raise ValueError("collector cache local_parts must be a list of names")
    return cache


def _read_cache(path):
    with open(path, encoding="utf-8") as stream:
        return _validated_cache(json.load(stream))


def _cache_recovery_paths():
    directory = os.path.dirname(CACHE_PATH) or "."
    basename = os.path.basename(CACHE_PATH)
    candidates = set(glob.glob(CACHE_PATH + ".tmp*"))
    candidates.update(glob.glob(os.path.join(
        directory, f".{basename}.*.tmp")))
    candidates.discard(CACHE_PATH)

    def modified(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1.0

    return sorted(candidates, key=lambda path: (modified(path), path), reverse=True)


def _load_cache():
    canonical_missing = False
    canonical_error = None
    try:
        # Mounted-drive directory metadata can lag behind direct path access.
        # Always attempt the canonical open before consulting recovery files.
        return _read_cache(CACHE_PATH)
    except FileNotFoundError:
        canonical_missing = True
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        canonical_error = exc

    recovery_paths = _cache_recovery_paths()
    recovery_errors = []
    for path in recovery_paths:
        try:
            return _read_cache(path)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            recovery_errors.append(f"{os.path.basename(path)}: {exc}")

    master_path = os.path.join(DATASET_DIR, "train.parquet")
    try:
        with open(master_path, "rb"):
            master_exists = True
    except FileNotFoundError:
        master_exists = False
    except OSError as exc:
        raise RuntimeError(
            f"cannot determine whether collector dataset exists: {exc}") from exc
    if canonical_missing and not recovery_paths and not master_exists:
        return _empty_cache()

    details = []
    if canonical_error is not None:
        details.append(f"canonical unreadable: {canonical_error}")
    elif canonical_missing:
        details.append("canonical missing")
    if recovery_errors:
        details.append("recovery invalid: " + "; ".join(recovery_errors))
    elif recovery_paths:
        details.append("no valid recovery cache")
    raise RuntimeError("collector cache unavailable; " + "; ".join(details))


def _write_cache_bytes_verified(path, encoded, attempts=CACHE_WRITE_ATTEMPTS):
    last_error = None
    for attempt in range(attempts):
        try:
            with open(path, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            last_error = exc
        try:
            with open(path, "rb") as stream:
                if stream.read() == encoded:
                    return
        except OSError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(CACHE_RETRY_SECONDS)
    raise RuntimeError(f"verified collector cache write failed for {path}: {last_error}")


def _save_cache(c):
    cache = _validated_cache(c)
    encoded = (json.dumps(
        cache, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    os.makedirs(DATASET_DIR, exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(CACHE_PATH)}.", suffix=".tmp",
        dir=os.path.dirname(CACHE_PATH) or ".")
    os.close(fd)
    staged_valid = False
    committed = False
    try:
        _write_cache_bytes_verified(staged, encoded)
        staged_valid = True
        last_replace_error = None
        for attempt in range(CACHE_WRITE_ATTEMPTS):
            try:
                os.replace(staged, CACHE_PATH)
            except OSError as exc:
                last_replace_error = exc
            try:
                with open(CACHE_PATH, "rb") as stream:
                    if stream.read() == encoded:
                        committed = True
                        return
            except OSError as exc:
                last_replace_error = exc
            if attempt + 1 < CACHE_WRITE_ATTEMPTS:
                time.sleep(CACHE_RETRY_SECONDS)

        # RaiDrive can deny replace even though direct overwrite is available.
        # The train lock serializes collector writers; keep the validated stage
        # until the canonical direct write has also passed an exact readback.
        try:
            _write_cache_bytes_verified(CACHE_PATH, encoded)
        except RuntimeError as exc:
            raise RuntimeError(
                "collector cache replace failed "
                f"({last_replace_error}); valid recovery remains at {staged}: {exc}"
            ) from exc
        committed = True
    finally:
        if (committed or not staged_valid) and os.path.exists(staged):
            try:
                os.remove(staged)
            except OSError:
                pass


def _commit_pending_cache(pending_harvested, pending_nodata, pending_local_parts=()):
    """Merge pending terminal IDs into the latest on-disk cache."""
    if not pending_harvested and not pending_nodata and not pending_local_parts:
        return
    cache = _load_cache()
    harvested = cache.setdefault("harvested", [])
    nodata = cache.setdefault("nodata", [])
    local_parts = cache.setdefault("local_parts", [])
    harvested.extend(tid for tid in pending_harvested if tid not in harvested)
    nodata.extend(tid for tid in pending_nodata if tid not in nodata)
    known_local_parts = set(local_parts)
    for part in pending_local_parts:
        if part not in known_local_parts:
            local_parts.append(part)
            known_local_parts.add(part)
    _save_cache(cache)


def fetch_result_rows(task_id, out=None):
    if out is None:
        out = fetch_stdout(task_id)
    if "===RESULT_CSV===" not in out:
        return None
    block = out.split("===RESULT_CSV===")[-1].split("===FAILED_CSV===")[0].strip()
    if not block or "," not in block:
        return None
    try:
        return pd.read_csv(io.StringIO(block))
    except Exception:
        return None


# 프로브 전치 버그 수정 커밋 (2026-07-07). 이전 코드로 돌린 행은 _side/core_center
# 프로브가 전치된 시트에서 평가된 값이라 무효 -> NaN 처리 (T_max_*, leeward는 유효)
PROBE_FIX_HASHES_OK = None  # lazy: revision -> merge-base ancestry result
PROBE_FIX_COMMIT = "6245ae84ba2734d2f1b6619fba3b2a8f15d20f42"
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")



def fetch_streamed_rows(task_id, out=None):
    """stdout의 RESULT_JSON 라인들 -> DataFrame (새 공유폴더 방식의 기본 회수 경로)"""
    if out is None:
        out = fetch_stdout(task_id)
    rows = []
    for l in out.splitlines():
        if l.startswith("RESULT_JSON "):
            try:
                rows.append(json.loads(l[12:]))
            except Exception:
                pass
    return pd.DataFrame(rows) if rows else None


def guard_terminal_result_json(frame, task_status):
    """Validate failed/cancelled RESULT_JSON without ever promoting a flag."""
    status = str(task_status or "").strip().lower()
    if frame is None or frame.empty or status not in {"failed", "cancelled"}:
        return frame, 0

    regression_root = os.path.abspath(os.path.join(HERE, ".."))
    if regression_root not in sys.path:
        sys.path.insert(0, regression_root)
    from quality_contract import annotate_validity

    audited = annotate_validity(frame)
    clean = frame.copy()
    strict_em = audited["_strict_valid_em"].fillna(False).astype(bool)
    strict_full = audited["_strict_valid_full"].fillna(False).astype(bool)
    claimed_em = (
        pd.to_numeric(clean["result_valid_em"], errors="coerce").eq(1)
        if "result_valid_em" in clean.columns
        else pd.Series(False, index=clean.index)
    )
    claimed_thermal = (
        pd.to_numeric(clean["result_valid_thermal"], errors="coerce").eq(1)
        if "result_valid_thermal" in clean.columns
        else pd.Series(False, index=clean.index)
    )
    if "result_valid_em" in clean.columns:
        clean.loc[claimed_em & ~strict_em, "result_valid_em"] = 0
    if "result_valid_thermal" in clean.columns:
        # Thermal recovery also requires clean/profile-valid EM provenance;
        # _strict_valid_thermal alone intentionally does not include it.
        clean.loc[claimed_thermal & ~strict_full, "result_valid_thermal"] = 0

    recoverable = claimed_em & strict_em & (~claimed_thermal | strict_full)
    clean["collector_terminal_status"] = status
    clean["terminal_result_recovery_validated"] = recoverable.astype(int)
    clean["terminal_result_recovery_reason"] = audited[
        "_strict_invalid_reasons"
    ].astype(str)
    return clean, int(recoverable.sum())


def _tag_source(frame, rank):
    tagged = frame.copy()
    tagged[SOURCE_RANK_COLUMN] = rank
    return tagged


def _deduplicate_ranked_rows(frame, dedup_keys):
    """Deduplicate after ordering rows by their durable source precedence."""
    if SOURCE_RANK_COLUMN in frame.columns:
        frame = frame.sort_values(SOURCE_RANK_COLUMN, kind="stable")
    return deduplicate_rows(frame, dedup_keys)


def load_local_result_frames(cache):
    """Load the current CSV and only parquet parts not committed previously."""
    frames = []
    pending_parts = []
    read_errors = 0

    if os.path.isfile(LOCAL_RESULTS_CSV):
        try:
            # The producer uses the same lock for append and schema rotation.
            # saved_at/project_name are its final columns, so requiring both also
            # rejects a row left truncated by an interrupted append.
            with FileLock(LOCAL_RESULTS_CSV + ".lock"):
                frame = pd.read_csv(LOCAL_RESULTS_CSV, on_bad_lines="skip")
            complete = _complete_key_mask(frame, ["project_name", "saved_at"])
            incomplete_count = int((~complete).sum())
            if incomplete_count:
                print(f"local csv incomplete rows skipped: {incomplete_count}")
                frame = frame.loc[complete].reset_index(drop=True)
            if len(frame):
                frames.append(_tag_source(frame, SOURCE_RANK_LOCAL_CSV))
                print(f"local csv rows: {len(frame)}")
        except Exception as exc:
            read_errors += 1
            print(f"local csv read failed: {exc}")

    cached_parts = set(cache.get("local_parts", []))
    pattern = os.path.join(LOCAL_RESULTS_PARTS_DIR, "*.parquet")
    for path in sorted(glob.glob(pattern)):
        part_id = os.path.basename(path)
        if part_id in cached_parts:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            # A part can be visible before its parquet footer is complete. Leave it
            # uncached so a later collector pass retries it.
            read_errors += 1
            print(f"local parquet read failed ({part_id}): {exc}")
            continue
        pending_parts.append(part_id)
        if len(frame):
            frames.append(_tag_source(frame, SOURCE_RANK_LOCAL_PART))

    if pending_parts:
        print(f"local parquet parts: {len(pending_parts)} new files")
    if read_errors:
        print(f"local read errors: {read_errors} (left uncached for retry)")
    return frames, pending_parts


def _probe_fix_git_run(command):
    """Run a read-only git query against the solver repository."""
    import subprocess

    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.join(HERE, "..", ".."),
    )


def _resolve_git_commit(revision):
    result = _probe_fix_git_run(
        ["git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"]
    )
    resolved = str(result.stdout or "").strip().lower()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        return None
    return resolved


def _initialize_probe_fix_ancestry_cache():
    """Validate the local repo before trusting any cached ancestry result."""
    global PROBE_FIX_HASHES_OK
    if PROBE_FIX_HASHES_OK is not None:
        return
    try:
        baseline = _resolve_git_commit(PROBE_FIX_COMMIT)
    except Exception as exc:
        raise RuntimeError("probe-fix git ancestry classification failed") from exc
    if baseline is None:
        raise RuntimeError(
            "probe-fix git ancestry classification failed: baseline commit unavailable"
        )
    PROBE_FIX_HASHES_OK = {
        PROBE_FIX_COMMIT: True,
        baseline: True,
    }


def _is_post_probe_fix_revision(value):
    """Classify a solver revision by its actual graph ancestry.

    The old implementation enumerated only the ancestry path from the fix to
    the collector's current HEAD. Valid commits on another branch were thus
    incorrectly treated as pre-fix. Resolve each observed revision and ask git
    whether the fix is its ancestor instead. Unknown or malformed hashes are
    untrusted, while an unavailable repository aborts classification.
    """
    global PROBE_FIX_HASHES_OK
    _initialize_probe_fix_ancestry_cache()

    if not isinstance(value, str):
        return False
    revision = value.strip().lower()
    if not GIT_REVISION_RE.fullmatch(revision):
        return False
    if revision in PROBE_FIX_HASHES_OK:
        return bool(PROBE_FIX_HASHES_OK[revision])

    try:
        resolved = _resolve_git_commit(revision)
        if resolved is None:
            # A syntactically valid but absent/ambiguous object is not trusted.
            PROBE_FIX_HASHES_OK[revision] = False
            return False
        if resolved in PROBE_FIX_HASHES_OK:
            verdict = bool(PROBE_FIX_HASHES_OK[resolved])
        else:
            result = _probe_fix_git_run(
                ["git", "merge-base", "--is-ancestor", PROBE_FIX_COMMIT, resolved]
            )
            if result.returncode == 0:
                verdict = True
            elif result.returncode == 1:
                verdict = False
            else:
                raise RuntimeError(
                    "git merge-base failed: " + str(result.stderr or "").strip()
                )
            PROBE_FIX_HASHES_OK[resolved] = verdict
        PROBE_FIX_HASHES_OK[revision] = verdict
        return verdict
    except Exception as exc:
        raise RuntimeError("probe-fix git ancestry classification failed") from exc


def sanitize_bad_probes(df):
    if "git_hash" not in df.columns:
        return df, 0
    # Initialize before changing the frame so a missing/wrong repository is a
    # fail-closed collector error, not a destructive mass demotion.
    _initialize_probe_fix_ancestry_cache()
    bad_cols = [
        column for column in df.columns
        if column.startswith("Tprobe_")
        and (
            column.endswith("_side_max")
            or column.endswith("_side_mean")
            or column.startswith("Tprobe_core_center_")
        )
    ]
    mask = ~df["git_hash"].map(_is_post_probe_fix_revision)
    n = int(mask.sum())
    if n and bad_cols:
        df.loc[mask, bad_cols] = float("nan")
    return df, n


def convergence_filter(df, max_err=1.5):
    """Keep the audit stream, but demote forged/legacy EM success flags.

    Partial RESULT_JSON rows remain valuable operational evidence and the
    collector historically preserves rows with no convergence columns.  They
    are never admitted to strict training: any explicit success flag is
    recomputed and demoted here, and ``quality_contract`` builds the actual
    strict cohort later.
    """
    regression_root = os.path.abspath(os.path.join(HERE, ".."))
    if regression_root not in sys.path:
        sys.path.insert(0, regression_root)
    from quality_contract import annotate_validity

    audited = annotate_validity(df)
    clean = audited.copy()
    if "result_valid_em" in clean.columns:
        claimed = pd.to_numeric(clean["result_valid_em"], errors="coerce").eq(1)
        invalid_claim = claimed & ~clean["_strict_valid_em"].fillna(False).astype(bool)
        clean.loc[invalid_claim, "result_valid_em"] = 0
        clean.loc[invalid_claim, "em_validity_reason"] = clean.loc[
            invalid_claim, "_strict_invalid_reasons"
        ]
    keep = pd.Series(True, index=clean.index)
    for column in ("conv_error_pct_matrix", "conv_error_pct_loss"):
        if column in clean.columns:
            values = pd.to_numeric(clean[column], errors="coerce")
            keep &= values.isna() | values.le(max_err)
    clean = clean.loc[keep].drop(
        columns=[
            "_strict_valid_em", "_strict_valid_thermal",
            "_strict_valid_full", "_strict_invalid_reasons",
        ],
        errors="ignore",
    )
    return clean, int((~keep).sum())


def normalize_thermal_validity(df):
    """Demote thermal success flags without complete physical and convergence evidence."""
    if "thermal_solved" not in df.columns:
        return df, 0
    solved = pd.to_numeric(df["thermal_solved"], errors="coerce").eq(1)
    if not solved.any():
        return df, 0

    complete = pd.Series(True, index=df.index)
    for column in MANDATORY_THERMAL_TEMPERATURE_COLUMNS:
        if column not in df.columns:
            complete &= False
        else:
            temperatures = pd.to_numeric(df[column], errors="coerce")
            complete &= (
                temperatures.map(math.isfinite)
                & temperatures.gt(MIN_TRUSTED_TEMPERATURE_C)
                & temperatures.lt(MAX_TRUSTED_TEMPERATURE_C)
            )

    if "N2_side" not in df.columns:
        complete &= False
        side_required = pd.Series(False, index=df.index)
    else:
        n2_side = pd.to_numeric(df["N2_side"], errors="coerce")
        complete &= n2_side.notna()
        side_required = n2_side.gt(0)
    physics_revision = df.get(
        "physics_data_revision", pd.Series("", index=df.index)
    ).fillna("").astype(str).str.strip()
    new_probe_required = side_required & physics_revision.eq(
        PHYSICS_DATA_REVISION
    )
    for column in SIDE_THERMAL_TEMPERATURE_COLUMNS:
        if column not in df.columns:
            complete &= ~side_required
        else:
            side_temperature = pd.to_numeric(df[column], errors="coerce")
            side_trusted = (
                side_temperature.map(math.isfinite)
                & side_temperature.gt(MIN_TRUSTED_TEMPERATURE_C)
                & side_temperature.lt(MAX_TRUSTED_TEMPERATURE_C)
            )
            complete &= ~side_required | side_trusted
    for column in STACKING_REVISION_SIDE_THERMAL_TEMPERATURE_COLUMNS:
        if column not in df.columns:
            complete &= ~new_probe_required
        else:
            side_temperature = pd.to_numeric(df[column], errors="coerce")
            side_trusted = (
                side_temperature.map(math.isfinite)
                & side_temperature.gt(MIN_TRUSTED_TEMPERATURE_C)
                & side_temperature.lt(MAX_TRUSTED_TEMPERATURE_C)
            )
            complete &= ~new_probe_required | side_trusted

    for column, expected in (
        ("thermal_rx_side_probe_contract_version",
         RX_SIDE_FACE_PROBE_CONTRACT_VERSION),
        ("thermal_rx_side_probe_max_rule", RX_SIDE_FACE_MAX_RULE),
        ("thermal_rx_side_probe_mean_rule", RX_SIDE_FACE_MEAN_RULE),
    ):
        if column not in df.columns:
            complete &= ~new_probe_required
        else:
            complete &= ~new_probe_required | df[column].fillna("").astype(str).eq(
                expected
            )
    if "thermal_rx_side_probe_selected_face" not in df.columns:
        complete &= ~new_probe_required
    else:
        complete &= ~new_probe_required | df[
            "thermal_rx_side_probe_selected_face"
        ].fillna("").astype(str).isin({
            "Tprobe_Rx_side_side", "Tprobe_Rx_side1_inner",
            "Tprobe_Rx_side2_side", "Tprobe_Rx_side2_inner",
        })
    if "thermal_rx_side_probe_face_count" not in df.columns:
        complete &= ~new_probe_required
    else:
        mode = df.get(
            "thermal_symmetry", pd.Series("", index=df.index)
        ).fillna("").astype(str).str.strip().str.lower()
        expected_faces = pd.Series(2.0, index=df.index).where(
            ~mode.eq("full"), 4.0
        )
        face_count = pd.to_numeric(
            df["thermal_rx_side_probe_face_count"], errors="coerce"
        )
        complete &= ~new_probe_required | face_count.eq(expected_faces)

    expected_mask = pd.Series(11, index=df.index).where(~side_required, 15)
    for column, predicate in (
        ("thermal_required_group_mask", lambda values: values.eq(expected_mask)),
        ("thermal_required_missing_count", lambda values: values.eq(0)),
        ("thermal_extraction_complete", lambda values: values.eq(1)),
    ):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            has_contract = values.notna()
            complete &= ~has_contract | predicate(values)

    # New solver rows must carry native residual proof. Legacy rows without the
    # field remain EM-only rather than being silently trusted as thermal data.
    for column in ("thermal_convergence_available", "thermal_converged"):
        if column not in df.columns:
            complete &= False
        else:
            complete &= pd.to_numeric(df[column], errors="coerce").eq(1)
    residual_columns = (
        "thermal_residual_continuity",
        "thermal_residual_x_velocity",
        "thermal_residual_y_velocity",
        "thermal_residual_z_velocity",
    )
    if "thermal_residual_flow_limit" not in df.columns:
        complete &= False
    else:
        flow_limit = pd.to_numeric(df["thermal_residual_flow_limit"], errors="coerce")
        complete &= flow_limit.map(math.isfinite) & flow_limit.gt(0) & flow_limit.le(1e-3)
        for column in residual_columns:
            if column not in df.columns:
                complete &= False
            else:
                residual = pd.to_numeric(df[column], errors="coerce")
                complete &= residual.map(math.isfinite) & residual.ge(0) & residual.le(flow_limit)
    if "thermal_residual_energy_limit" not in df.columns or "thermal_residual_energy" not in df.columns:
        complete &= False
    else:
        energy_limit = pd.to_numeric(df["thermal_residual_energy_limit"], errors="coerce")
        energy = pd.to_numeric(df["thermal_residual_energy"], errors="coerce")
        complete &= (
            energy_limit.map(math.isfinite) & energy_limit.gt(0) & energy_limit.le(1e-7)
            & energy.map(math.isfinite) & energy.ge(0) & energy.le(energy_limit)
        )
    if "thermal_iterations" not in df.columns:
        complete &= False
    else:
        complete &= pd.to_numeric(df["thermal_iterations"], errors="coerce").gt(0)

    # Recompute the canonical contract as well.  The local checks above retain
    # compatibility with old audit columns; this adds the previously missing
    # power-balance and all-temperature saturation evidence.
    regression_root = os.path.abspath(os.path.join(HERE, ".."))
    if regression_root not in sys.path:
        sys.path.insert(0, regression_root)
    from quality_contract import annotate_validity

    audited = annotate_validity(df)
    complete &= audited["_strict_valid_thermal"].fillna(False).astype(bool)
    invalid = solved & ~complete
    count = int(invalid.sum())
    if count:
        df = df.copy()
        df.loc[invalid, "thermal_solved"] = 0
        if "result_valid_thermal" not in df.columns:
            df["result_valid_thermal"] = float("nan")
        df.loc[invalid, "result_valid_thermal"] = 0
    return df, count


def reject_explicit_dirty_provenance(df):
    """Drop rows that explicitly report dirty or untrusted execution provenance."""
    if df is None or df.empty:
        return df, 0
    keep = pd.Series(True, index=df.index)
    for column in ("git_dirty", "pyaedt_library_git_dirty"):
        if column not in df.columns:
            continue
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        keep &= raw.isna() | numeric.eq(0)
    for column in ("matrix_solve_attempts", "loss_solve_attempts"):
        if column not in df.columns:
            continue
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        keep &= raw.isna() | numeric.eq(1)
    if "matrix_extraction_backend" in df.columns:
        raw = df["matrix_extraction_backend"]
        keep &= raw.isna() | raw.astype(str).eq("export_rl_matrix")
    rejected = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), rejected


def _row_hashes(df, columns):
    normalized = df.loc[:, columns].astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False)


def _fallback_columns(frame, other=None):
    """Columns used for legacy identity, excluding collector-added metadata."""
    excluded = {"task_id", "task_name", SOURCE_RANK_COLUMN}
    if other is None:
        return [c for c in frame.columns if c not in excluded]
    return [c for c in frame.columns if c in other.columns and c not in excluded]


def _complete_key_mask(frame, dedup_keys):
    if not dedup_keys or not all(c in frame.columns for c in dedup_keys):
        return pd.Series(False, index=frame.index)
    return frame.loc[:, dedup_keys].notna().all(axis=1)


def deduplicate_rows(frame, dedup_keys):
    """Dedup complete keys by key and incomplete keys by stable row content."""
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame

    complete = _complete_key_mask(frame, dedup_keys)
    keep = pd.Series(True, index=frame.index)
    if complete.any():
        keyed_duplicates = frame.loc[complete, dedup_keys].duplicated(keep="last")
        keep.loc[complete] = ~keyed_duplicates.to_numpy()

    incomplete = ~complete
    if incomplete.any():
        fallback_columns = _fallback_columns(frame)
        if fallback_columns:
            hashes = _row_hashes(frame.loc[incomplete], fallback_columns)
            keep.loc[incomplete] = ~hashes.duplicated(keep="last").to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def select_new_unique_rows(incoming, old, dedup_keys):
    """Select absent identities and higher-ranked replacements from the master."""
    incoming = _deduplicate_ranked_rows(incoming, dedup_keys)
    if old is None or old.empty:
        return incoming.reset_index(drop=True)

    new_mask = pd.Series(True, index=incoming.index)
    can_compare_keys = bool(dedup_keys) and all(c in old.columns for c in dedup_keys)
    incoming_complete = _complete_key_mask(incoming, dedup_keys)
    if can_compare_keys and incoming_complete.any():
        old_complete = _complete_key_mask(old, dedup_keys)
        old_key_hash_series = _row_hashes(old.loc[old_complete], dedup_keys)
        old_key_hashes = set(old_key_hash_series.tolist())
        incoming_key_hashes = _row_hashes(
            incoming.loc[incoming_complete], dedup_keys)
        accepted = ~incoming_key_hashes.isin(old_key_hashes)
        if SOURCE_RANK_COLUMN in incoming.columns:
            if SOURCE_RANK_COLUMN in old.columns:
                old_ranks = pd.to_numeric(
                    old.loc[old_complete, SOURCE_RANK_COLUMN], errors="coerce"
                ).fillna(0)
            else:
                old_ranks = pd.Series(0, index=old_key_hash_series.index, dtype=float)
            old_rank_by_hash = {}
            for key_hash, rank in zip(old_key_hash_series.tolist(), old_ranks.tolist()):
                old_rank_by_hash[key_hash] = max(old_rank_by_hash.get(key_hash, 0), rank)
            incoming_ranks = pd.to_numeric(
                incoming.loc[incoming_complete, SOURCE_RANK_COLUMN], errors="coerce"
            ).fillna(0)
            higher_rank = [
                rank > old_rank_by_hash.get(key_hash, float("inf"))
                for key_hash, rank in zip(incoming_key_hashes.tolist(), incoming_ranks.tolist())
            ]
            accepted |= pd.Series(higher_rank, index=accepted.index)
        new_mask.loc[incoming_complete] = accepted.to_numpy()

    fallback_mask = ~incoming_complete if can_compare_keys else pd.Series(
        True, index=incoming.index)
    if fallback_mask.any():
        fallback_columns = _fallback_columns(incoming, old)
        if fallback_columns:
            old_hashes = set(_row_hashes(old, fallback_columns).tolist())
            incoming_hashes = _row_hashes(
                incoming.loc[fallback_mask], fallback_columns)
            new_mask.loc[fallback_mask] = ~incoming_hashes.isin(
                old_hashes).to_numpy()
    return incoming.loc[new_mask].reset_index(drop=True)


def _drop_replaced_key_rows(old, replacements, dedup_keys):
    """Remove old complete-key rows superseded by accepted incoming rows."""
    if old is None or old.empty:
        return old
    old_complete = _complete_key_mask(old, dedup_keys)
    replacement_complete = _complete_key_mask(replacements, dedup_keys)
    if not old_complete.any() or not replacement_complete.any():
        return old
    replacement_hashes = set(
        _row_hashes(replacements.loc[replacement_complete], dedup_keys).tolist())
    replaced = pd.Series(False, index=old.index)
    replaced.loc[old_complete] = _row_hashes(
        old.loc[old_complete], dedup_keys).isin(replacement_hashes).to_numpy()
    return old.loc[~replaced].reset_index(drop=True)


def _source_rank_rows(frame, dedup_keys):
    """Return one numeric source-rank row per complete sample identity."""
    columns = list(dedup_keys) + [SOURCE_RANK_COLUMN]
    if frame is None or frame.empty or not all(c in frame.columns for c in dedup_keys):
        return pd.DataFrame(columns=columns)
    complete = _complete_key_mask(frame, dedup_keys)
    if not complete.any():
        return pd.DataFrame(columns=columns)
    ranked = frame.loc[complete, list(dedup_keys)].copy()
    if SOURCE_RANK_COLUMN in frame.columns:
        ranked[SOURCE_RANK_COLUMN] = pd.to_numeric(
            frame.loc[complete, SOURCE_RANK_COLUMN], errors="coerce").fillna(0).to_numpy()
    else:
        ranked[SOURCE_RANK_COLUMN] = 0
    return _deduplicate_ranked_rows(ranked, dedup_keys).reset_index(drop=True)


def _attach_source_ranks(frame, sidecar, dedup_keys):
    """Attach sidecar provenance for comparison without exposing it to training."""
    if frame is None:
        return None
    ranked = frame.copy()
    if SOURCE_RANK_COLUMN in ranked.columns:
        legacy_rank = pd.to_numeric(ranked[SOURCE_RANK_COLUMN], errors="coerce").fillna(0)
        ranked = ranked.drop(columns=[SOURCE_RANK_COLUMN])
    else:
        legacy_rank = pd.Series(0, index=ranked.index, dtype=float)
    ranked["_collector_legacy_rank"] = legacy_rank.to_numpy()

    sidecar = _source_rank_rows(sidecar, dedup_keys)
    if len(sidecar) and all(c in ranked.columns for c in dedup_keys):
        sidecar = sidecar.rename(columns={SOURCE_RANK_COLUMN: "_collector_saved_rank"})
        ranked = ranked.merge(sidecar, on=list(dedup_keys), how="left", sort=False)
        saved_rank = pd.to_numeric(
            ranked.pop("_collector_saved_rank"), errors="coerce").fillna(0)
    else:
        saved_rank = pd.Series(0, index=ranked.index, dtype=float)
    legacy_rank = pd.to_numeric(
        ranked.pop("_collector_legacy_rank"), errors="coerce").fillna(0)
    ranked[SOURCE_RANK_COLUMN] = pd.concat(
        [legacy_rank, saved_rank], axis=1).max(axis=1)
    return ranked


def _matching_replay_rows(master, incoming, identities, dedup_keys):
    """Return deterministic replay rows only when their persisted payload matches."""
    if incoming is None:
        raise RuntimeError(
            "source rank sidecar does not cover every master dataset identity")
    replay = _deduplicate_ranked_rows(incoming.copy(), dedup_keys)
    try:
        persisted = identities.merge(
            master, on=dedup_keys, how="left", validate="one_to_one")
        replayed = identities.merge(
            replay, on=dedup_keys, how="left", validate="one_to_one",
            indicator="_replay_merge")
    except (pd.errors.MergeError, ValueError) as exc:
        raise RuntimeError("replay payload identity is ambiguous") from exc
    if not replayed.pop("_replay_merge").eq("both").all():
        raise RuntimeError(
            "source rank sidecar does not cover every master dataset identity")

    excluded = {"task_id", "task_name", SOURCE_RANK_COLUMN}
    payload_columns = sorted(
        (set(persisted.columns) | set(replayed.columns)) - excluded)
    persisted = persisted.reindex(columns=payload_columns)
    replayed = replayed.reindex(columns=payload_columns)
    try:
        pd.testing.assert_frame_equal(
            persisted.reset_index(drop=True), replayed.reset_index(drop=True),
            check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise RuntimeError(
            "replay payload differs from persisted master") from exc
    return replay


def _validated_source_rank_sidecar(master, sidecar, incoming, dedup_keys):
    """Validate rank coverage, repairing only rows replayed after master-first install."""
    columns = list(dedup_keys) + [SOURCE_RANK_COLUMN]
    if sidecar is None:
        validated = pd.DataFrame(columns=columns)
    else:
        missing_columns = [column for column in columns if column not in sidecar.columns]
        if missing_columns:
            raise RuntimeError(
                f"source rank sidecar schema is invalid; missing {missing_columns}")
        validated = sidecar[columns].copy()
        complete = _complete_key_mask(validated, dedup_keys)
        ranks = pd.to_numeric(validated[SOURCE_RANK_COLUMN], errors="coerce")
        if (not complete.all() or not ranks.map(math.isfinite).all()
                or (ranks < 0).any() or validated.duplicated(dedup_keys).any()):
            raise RuntimeError("source rank sidecar contains invalid or duplicate rows")
        validated[SOURCE_RANK_COLUMN] = ranks

    if master is None or master.empty:
        return validated
    if not all(key in master.columns for key in dedup_keys):
        raise RuntimeError("master dataset is missing source-rank identity columns")
    identities = master.loc[
        _complete_key_mask(master, dedup_keys), dedup_keys].drop_duplicates()
    coverage = identities.merge(
        validated[dedup_keys], on=dedup_keys, how="left", indicator=True)
    missing = coverage.loc[coverage["_merge"].ne("both"), dedup_keys]
    if not len(missing):
        return validated

    replay = _matching_replay_rows(master, incoming, missing, dedup_keys)
    replay_ranks = _source_rank_rows(replay, dedup_keys)
    repairable = missing.merge(
        replay_ranks, on=dedup_keys, how="left", indicator=True)
    if (not repairable["_merge"].eq("both").all()
            or not pd.to_numeric(
                repairable[SOURCE_RANK_COLUMN], errors="coerce").map(math.isfinite).all()):
        raise RuntimeError(
            "source rank sidecar does not cover every master dataset identity")
    repaired = repairable[columns]
    return pd.concat([validated, repaired], ignore_index=True, sort=False)


def _stage_path(target):
    fd, path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=os.path.dirname(target))
    os.close(fd)
    return path


def _stage_parquet(frame, target):
    staged = _stage_path(target)
    try:
        frame.to_parquet(staged, index=False)
        return staged
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise


def _stage_csv(frame, target):
    staged = _stage_path(target)
    try:
        frame.to_csv(staged, index=False)
        return staged
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise


def _stage_train_io_views(master):
    """Serialize both curated views before either target is replaced."""
    io_frame = build_train_io(master)
    parquet_path = os.path.join(DATASET_DIR, "train_io.parquet")
    csv_path = os.path.join(DATASET_DIR, "train_io.csv")
    staged = []
    try:
        staged.append((_stage_parquet(io_frame, parquet_path), parquet_path))
        staged.append((_stage_csv(io_frame, csv_path), csv_path))
        return staged
    except Exception:
        for staged_path, _ in staged:
            if os.path.exists(staged_path):
                os.remove(staged_path)
        raise


def _replace_train_io_views(master):
    """Atomically refresh each curated view. Caller must hold the dataset lock."""
    _replace_staged(_stage_train_io_views(master))


def _stage_manifest(manifest, target):
    staged = _stage_path(target)
    try:
        with open(staged, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        return staged
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise


def _replace_staged(staged_targets):
    """Install fully serialized files in the caller's recovery-safe order."""
    try:
        for staged, target in staged_targets:
            os.replace(staged, target)
    finally:
        for staged, _ in staged_targets:
            if os.path.exists(staged):
                os.remove(staged)


def repair_master_thermal_validity():
    """Atomically demote thermal rows that predate the current physical gate."""
    master_path = os.path.join(DATASET_DIR, "train.parquet")
    if not os.path.isfile(master_path):
        return 0
    lock_path = master_path + ".lock"
    with FileLock(lock_path):
        frame = pd.read_parquet(master_path)
        repaired, count = normalize_thermal_validity(frame)
        if not count:
            return 0
        staged = _stage_parquet(repaired, master_path)
        try:
            os.replace(staged, master_path)
        finally:
            if os.path.exists(staged):
                os.remove(staged)
        return count


def merge_dataset(merged, dedup_keys, prefix, pending_harvested=(), pending_nodata=(),
                  pending_local_parts=()):
    """Merge rows and terminal cache state under one inter-process lock."""
    master_path = os.path.join(DATASET_DIR, "train.parquet")
    manifest_path = os.path.join(DATASET_DIR, "manifest.json")
    source_rank_path = os.path.join(DATASET_DIR, "source_ranks.parquet")
    lock_path = master_path + ".lock"
    with FileLock(lock_path):
        old = pd.read_parquet(master_path) if os.path.isfile(master_path) else None
        source_ranks = (
            pd.read_parquet(source_rank_path) if os.path.isfile(source_rank_path) else None
        )
        source_ranks = _validated_source_rank_sidecar(
            old, source_ranks, merged, dedup_keys)
        ranked_old = _attach_source_ranks(old, source_ranks, dedup_keys)
        new_rows = select_new_unique_rows(merged, ranked_old, dedup_keys)
        new_unique_rows = len(new_rows)
        if not new_unique_rows:
            _replace_train_io_views(old if old is not None else pd.DataFrame())
            _commit_pending_cache(
                pending_harvested, pending_nodata, pending_local_parts)
            return new_unique_rows, 0 if old is None else len(old), master_path

        stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        part_path = os.path.join(
            DATASET_DIR, f"collected_{prefix.replace('/', '_')}_{stamp}.parquet")
        retained_old = _drop_replaced_key_rows(ranked_old, new_rows, dedup_keys)
        ranked_allf = new_rows if retained_old is None else pd.concat(
            [retained_old, new_rows], ignore_index=True, sort=False)
        source_ranks = _source_rank_rows(ranked_allf, dedup_keys)
        persisted_new_rows = new_rows.drop(columns=[SOURCE_RANK_COLUMN], errors="ignore")
        allf = ranked_allf.drop(columns=[SOURCE_RANK_COLUMN], errors="ignore")
        manifest = {
            "updated": stamp, "total_rows": len(allf), "new_rows": new_unique_rows,
            "new_unique_rows": new_unique_rows,
            "git_hashes": sorted(allf["git_hash"].dropna().unique().tolist())
            if "git_hash" in allf.columns else [],
            "prefix": prefix,
        }

        staged = []
        try:
            staged.append((_stage_parquet(persisted_new_rows, part_path), part_path))
            staged.append((_stage_manifest(manifest, manifest_path), manifest_path))
            staged.append((_stage_parquet(allf, master_path), master_path))
            # Rank follows master. If this replacement fails, the part remains
            # uncached and the lower-rank sidecar causes a safe retry next pass.
            staged.append((_stage_parquet(source_ranks, source_rank_path), source_rank_path))
            staged.extend(_stage_train_io_views(allf))
            _replace_staged(staged)
        except Exception:
            for temp_path, _ in staged:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise

        _commit_pending_cache(
            pending_harvested, pending_nodata, pending_local_parts)
        return new_unique_rows, len(allf), master_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mft-camp")
    ap.add_argument(
        "--extra-prefix", action="append", default=[],
        help=(
            "additional scheduler task-name prefix to collect; repeat this "
            "option to use an explicit allowlist"
        ),
    )
    ap.add_argument("--max-conv-err", type=float, default=1.5)
    ap.add_argument("--cancelled-fetch-limit", type=int, default=500)
    ap.add_argument(
        "--running-fetch-limit", type=int, default=32,
        help=(
            "maximum running/attaching stdout files to inspect for early "
            "RESULT_JSON rows; terminal collection is unaffected"
        ),
    )
    ap.add_argument(
        "--recollect-task", action="append", type=int, default=[],
        help=(
            "force a terminal scheduler task through the current collector "
            "policy; repeatable and idempotent by sample identity"
        ),
    )
    args = ap.parse_args(argv)

    recollect_task_ids = set(args.recollect_task)
    if any(task_id <= 0 for task_id in recollect_task_ids):
        ap.error("--recollect-task IDs must be positive")

    os.makedirs(DATASET_DIR, exist_ok=True)
    repaired_master_rows = repair_master_thermal_validity()
    if repaired_master_rows:
        print(f"thermal_master_rows_demoted: {repaired_master_rows}")

    prefixes = tuple(dict.fromkeys([args.prefix, *args.extra_prefix]))
    tasks = list_tasks_for_prefixes(prefixes)
    tasks_by_id = {int(task["id"]): task for task in tasks}
    for task_id in sorted(recollect_task_ids - set(tasks_by_id)):
        task = _get_json(f"/api/tasks/{task_id}", timeout=15)
        if not isinstance(task, dict) or int(task.get("id", 0) or 0) != task_id:
            raise RuntimeError(f"recollect task {task_id} lookup returned a mismatched task")
        if not any(str(task.get("name", "")).startswith(prefix) for prefix in prefixes):
            raise RuntimeError(
                f"recollect task {task_id} is outside the configured prefix allowlist"
            )
        tasks_by_id[task_id] = task
    tasks = sorted(tasks_by_id.values(), key=lambda task: int(task["id"]))
    nonterminal_recollect = sorted(
        task_id for task_id in recollect_task_ids
        if str(tasks_by_id[task_id].get("status")) not in {
            "completed", "failed", "cancelled"
        }
    )
    if nonterminal_recollect:
        raise RuntimeError(
            "recollect tasks must be terminal: "
            + ", ".join(str(task_id) for task_id in nonterminal_recollect)
        )
    collection_label = "+".join(prefixes)
    if len(prefixes) > 1:
        print(f"task prefixes: {', '.join(prefixes)}")
    done = [t for t in tasks if t.get("status") == "completed"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    cancelled = [t for t in tasks if t.get("status") == "cancelled"]
    status_counts = {}
    for task in tasks:
        status = str(task.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    active_count = sum(status_counts.get(status, 0) for status in ("queued", "attaching", "running"))
    print(
        f"tasks: {len(tasks)} (completed {len(done)}, failed {len(failed)}, "
        f"active {active_count}, cancelled {status_counts.get('cancelled', 0)})"
    )

    # 실패/취소 태스크도 stdout에 결과가 있으면 회수 (pyaedt teardown 크래시나
    # count 배치 중간 취소 전에 성공한 샘플이 남는 경우가 있음).
    # A bounded cancelled batch prevents historical backlog from monopolizing
    # the scheduler API; fresh completed/failed tasks are always handled first.
    # + 실행 중 태스크의 RESULT_JSON 라인도 스트리밍 회수 (샘플 단위 실시간성)
    import json as _json
    running = sorted(
        (t for t in tasks if t.get("status") in ("running", "attaching")),
        key=lambda task: int(task["id"]),
        reverse=True,
    )[:max(0, args.running_fetch_limit)]
    cache = _load_cache()
    skip = (
        set(cache.get("nodata", [])) | set(cache.get("harvested", []))
    ) - recollect_task_ids
    frames = []
    n_salvaged = 0
    n_terminal_recovered = 0
    n_streamed = 0
    n_skipped = 0
    n_fetch_errors = 0
    pending_harvested = []
    never_started_cancelled = [
        t for t in cancelled
        if t["id"] not in skip
        and "started_at" in t
        and t["started_at"] is None
    ]
    never_started_ids = {t["id"] for t in never_started_cancelled}
    pending_nodata = list(never_started_ids)
    pending_local_parts = []
    terminal = done + failed + cancelled
    n_skipped = sum(t["id"] in skip for t in terminal)
    pending_primary = sorted(
        (t for t in done + failed if t["id"] not in skip),
        key=lambda task: int(task["id"]), reverse=True,
    )
    pending_cancelled = sorted(
        (t for t in cancelled
         if t["id"] not in skip and t["id"] not in never_started_ids),
        key=lambda task: int(task["id"]),
    )[:max(0, args.cancelled_fetch_limit)]
    pending_terminal = pending_primary + pending_cancelled
    for t in pending_terminal:
        try:
            out = fetch_stdout(t["id"])
        except FetchError as exc:
            n_fetch_errors += 1
            print(f"fetch error task {t['id']}: {exc}")
            continue
        # Read both sources: JSON wins duplicate identities by rank, while the
        # legacy aggregate can still recover a row whose JSON line was truncated.
        task_frames = []
        json_frame = fetch_streamed_rows(t["id"], out=out)
        if json_frame is not None and len(json_frame):
            json_frame, recovered = guard_terminal_result_json(
                json_frame, t.get("status")
            )
            n_terminal_recovered += recovered
            json_rank = SOURCE_RANK_JSON + (
                SOURCE_RANK_RECOLLECT_OFFSET
                if int(t["id"]) in recollect_task_ids else 0
            )
            task_frames.append((json_frame, json_rank))
        csv_frame = fetch_result_rows(t["id"], out=out)
        if csv_frame is not None and len(csv_frame):
            csv_rank = SOURCE_RANK_TERMINAL_CSV + (
                SOURCE_RANK_RECOLLECT_OFFSET
                if int(t["id"]) in recollect_task_ids else 0
            )
            task_frames.append((csv_frame, csv_rank))
        if task_frames:
            for frame, source_rank in task_frames:
                frame["task_id"] = t["id"]
                frame["task_name"] = t.get("name", "")
                frames.append(_tag_source(frame, source_rank))
            pending_harvested.append(t["id"])
            if t.get("status") in ("failed", "cancelled"):
                n_salvaged += 1
        else:
            # Only a successful stdout fetch proves that a task has no data.
            pending_nodata.append(t["id"])
    if n_skipped:
        print(f"cache skip: {n_skipped} terminal tasks")
    if never_started_cancelled:
        print(
            f"never-started cancelled: {len(never_started_cancelled)} marked nodata")
    for t in running:
        try:
            out = fetch_stdout(t["id"], timeout=20)
        except FetchError as exc:
            n_fetch_errors += 1
            print(f"fetch error task {t['id']}: {exc}")
            continue
        rows = []
        for line in out.splitlines():
            if line.startswith("RESULT_JSON "):
                try:
                    rows.append(_json.loads(line[len("RESULT_JSON "):]))
                except Exception:
                    pass
        if rows:
            df = pd.DataFrame(rows)
            df["task_id"] = t["id"]
            df["task_name"] = t.get("name", "")
            frames.append(_tag_source(df, SOURCE_RANK_JSON))
            n_streamed += len(rows)
    if n_salvaged:
        print(f"salvaged from failed tasks: {n_salvaged}")
    if n_terminal_recovered:
        print(
            "strict terminal RESULT_JSON rows recovered: "
            f"{n_terminal_recovered}"
        )
    if n_streamed:
        print(f"streamed from running tasks: {n_streamed} rows")
    if n_fetch_errors:
        print(f"fetch_errors: {n_fetch_errors} (not cached as nodata)")

    # Local parquet parts are the durable schema-union source. The current CSV is
    # retained as a lower-ranked fallback when a parquet write failed.
    local_frames, pending_local_parts = load_local_result_frames(cache)
    for frame in local_frames:
        frame["task_name"] = "local"
        frames.append(frame)

    if not frames:
        master_path = os.path.join(DATASET_DIR, "train.parquet")
        with FileLock(master_path + ".lock"):
            master = (
                pd.read_parquet(master_path)
                if os.path.isfile(master_path) else pd.DataFrame()
            )
            _replace_train_io_views(master)
            _commit_pending_cache(
                pending_harvested, pending_nodata, pending_local_parts)
        print("new_unique_rows: 0")
        print("no result rows collected")
        return {"new_unique_rows": 0, "fetch_errors": n_fetch_errors}

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged, n_dirty = reject_explicit_dirty_provenance(merged)
    if n_dirty:
        print(f"provenance filter: {n_dirty} explicit dirty-source rows rejected")

    # 중복 제거 (재회수/재시도 대비)
    # Both fields are required for the primary identity. A missing column sends
    # every row through the content-hash legacy fallback.
    dedup_keys = ["project_name", "saved_at"]
    before = len(merged)
    merged = _deduplicate_ranked_rows(merged, dedup_keys)
    print(f"dedup: {before} -> {len(merged)}")

    merged, n_bad_probe = sanitize_bad_probes(merged)
    if n_bad_probe:
        print(f"probe-fix 이전 행 {n_bad_probe}개: side/core_center 프로브 컬럼 NaN 처리 (T_max/leeward는 유지)")
    merged, n_false_thermal = normalize_thermal_validity(merged)
    if n_false_thermal:
        print(
            f"thermal validity: {n_false_thermal} legacy false-success rows demoted to EM-only"
        )
    merged, n_filtered = convergence_filter(merged, args.max_conv_err)
    print(f"convergence filter (<= {args.max_conv_err}%): -{n_filtered} rows")

    new_unique_rows, total_rows, master_path = merge_dataset(
        merged, dedup_keys, collection_label,
        pending_harvested=pending_harvested, pending_nodata=pending_nodata,
        pending_local_parts=pending_local_parts)
    print(f"new_unique_rows: {new_unique_rows}")
    if not new_unique_rows:
        print(f"dataset unchanged: {total_rows} rows -> {master_path}")
        return {"new_unique_rows": 0, "fetch_errors": n_fetch_errors}
    print(f"dataset: {total_rows} rows total -> {master_path}")
    return {"new_unique_rows": new_unique_rows, "fetch_errors": n_fetch_errors}


if __name__ == "__main__":
    main()
