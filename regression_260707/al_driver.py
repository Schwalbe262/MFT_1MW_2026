"""
능동학습 검증 루프 드라이버 (재개 가능 상태기계).

라운드: TRAIN -> OPTIMIZE -> SELECT -> SUBMIT -> WAIT -> INGEST -> CHECK -> (다음 라운드 | 종료)

- TRAIN: train_models.py (고정 하이퍼파라미터 재학습, 검증 데이터 sample_weight=3)
- OPTIMIZE: run_nsga2.py (16 재시작, warm start, 밀도 게이트)
- SELECT: select_candidates.select (K=33: 활용/경계/탐사/재검증)
- SUBMIT/WAIT: scheduler_client (fea_bursty, RESULT_JSON 회수, 실패 시 64GB 1회 재시도)
- INGEST: 검증 rows -> dataset 병합 + 예측 vs 실측 오차 기록 -> q 적응
- CHECK: 종료판정 (스펙 통과 >=3 + 배치 일치도 + HV 정체) 또는 하드캡 10라운드

사용:
  python al_driver.py            # state.json 이 있으면 이어서
  python al_driver.py --reset    # 처음부터
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = HERE
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from filelock import FileLock
from module.source_contract import SOLVER_REVISION_PATHS

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "training"))
sys.path.insert(0, os.path.join(HERE, "verify"))

RUNTIME_ROOT = HERE
OUTPUT_ROOT = HERE
AL_ROOT = os.path.join(OUTPUT_ROOT, "al_rounds")
STATE_PATH = os.path.join(AL_ROOT, "state.json")
DATASET = os.path.join(RUNTIME_ROOT, "data", "dataset", "train.parquet")
REGISTRY = os.path.join(RUNTIME_ROOT, "training", "registry")
QUALITY_STATUS = os.path.join(RUNTIME_ROOT, "training", "model_quality_status.json")
EXECUTE_SUBMISSIONS = False
PINNED_SOLVER_REVISION = None
PINNED_LIBRARY_REVISION = None
PINNED_LIBRARY_ROOT = None
PY = sys.executable
MIN_STRICT_FULL_ROWS = 3000
QUALITY_THRESHOLDS_PATH = os.path.join(
    CODE_ROOT, "training", "model_quality_thresholds.json"
)

SPEC = {
    "Llt_target_uH": 27.5, "Llt_tol_uH": 0.55,
    "T_limit_C": 100.0, "B_limit_T": 1.2,
    "agree_llt_med_pct": 0.5, "agree_llt_max_pct": 1.0,
    "agree_T_med_C": 3.0, "agree_T_max_C": 5.0,
    "agree_P_med_pct": 3.0, "agree_P_max_pct": 5.0,
    "verification_min_coverage": 0.70,
    "convergence_max_pct": 1.5,
    "max_rounds": 10, "K": 33,
}

T_TARGETS = ["Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
             "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max"]

ACTUAL_TEMPERATURES = {
    "Tprobe_Tx_leeward_max": "T_max_Tx",
    "Tprobe_Rx_main_leeward_max": "T_max_Rx_main",
    "Tprobe_Rx_side_leeward_max": "T_max_Rx_side",
    "Tprobe_core_center_max": "T_max_core",
}
TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}
LOSS_COMPONENTS = (
    "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
)
SOURCE_RANK_COLUMN = "_collector_source_rank"
AL_SOURCE_RANK = 50


def configure_runtime(
        runtime_root=None, dataset=None, output_root=None, registry=None,
        solver_revision=None, library_revision=None, library_root=None):
    """Point code in this worktree at a live runtime without changing its HEAD."""
    global RUNTIME_ROOT, OUTPUT_ROOT, AL_ROOT, STATE_PATH, DATASET, REGISTRY, QUALITY_STATUS
    global PINNED_SOLVER_REVISION, PINNED_LIBRARY_REVISION, PINNED_LIBRARY_ROOT
    RUNTIME_ROOT = os.path.abspath(runtime_root or HERE)
    OUTPUT_ROOT = os.path.abspath(output_root or RUNTIME_ROOT)
    AL_ROOT = os.path.join(OUTPUT_ROOT, "al_rounds")
    STATE_PATH = os.path.join(AL_ROOT, "state.json")
    DATASET = os.path.abspath(
        dataset or os.path.join(RUNTIME_ROOT, "data", "dataset", "train.parquet")
    )
    REGISTRY = os.path.abspath(
        registry or os.path.join(RUNTIME_ROOT, "training", "registry")
    )
    QUALITY_STATUS = os.path.join(
        RUNTIME_ROOT, "training", "model_quality_status.json"
    )
    PINNED_SOLVER_REVISION = solver_revision
    PINNED_LIBRARY_REVISION = library_revision
    PINNED_LIBRARY_ROOT = os.path.abspath(library_root) if library_root else None


def _require_runtime_deployment():
    """Recheck both advertised remote heads immediately before any submit."""
    library_root = PINNED_LIBRARY_ROOT or os.environ.get(
        "MFT_PYAEDT_LIBRARY_ROOT", ""
    ).strip()
    if not library_root:
        raise RuntimeError(
            "AL submission requires --library-root or MFT_PYAEDT_LIBRARY_ROOT"
        )
    from campaign.deployment_gate import validate_deployment

    return validate_deployment(
        REPO_ROOT,
        _current_solver_revision(),
        library_root,
        _current_library_revision(),
    )


def _active_al_root():
    """Honor legacy tests/tools that temporarily redirect ``HERE``."""
    if RUNTIME_ROOT == CODE_ROOT and OUTPUT_ROOT == CODE_ROOT:
        return os.path.join(HERE, "al_rounds")
    return AL_ROOT


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity():
    if not PINNED_SOLVER_REVISION or not PINNED_LIBRARY_REVISION:
        raise RuntimeError("AL runtime requires explicit solver/library revision pins")
    return {
        "schema_version": 2,
        "solver_revision": str(PINNED_SOLVER_REVISION).lower(),
        "library_revision": str(PINNED_LIBRARY_REVISION).lower(),
        "dataset": os.path.abspath(DATASET),
        "registry": os.path.abspath(REGISTRY),
        "quality_thresholds_sha256": _sha256(QUALITY_THRESHOLDS_PATH),
        "minimum_strict_full_rows": MIN_STRICT_FULL_ROWS,
    }


def _bind_runtime_identity(state):
    expected = _runtime_identity()
    existing = state.get("runtime_identity")
    if existing is None:
        fresh_keys = {"round", "stage", "q_mult", "task_map", "history"}
        is_fresh = (
            set(state).issubset(fresh_keys)
            and int(state.get("round", -1)) == 1
            and state.get("stage") == "TRAIN"
            and float(state.get("q_mult", -1)) == 1.0
            and not state.get("training_run_id")
            and not state.get("task_map")
            and not state.get("history")
        )
        if not is_fresh:
            raise RuntimeError(
                "legacy/unpinned AL state cannot be resumed; archive it and use --reset"
            )
        state["runtime_identity"] = expected
    elif existing != expected:
        raise RuntimeError(
            f"AL runtime identity mismatch: stored={existing}, expected={expected}"
        )
    return expected


def _assert_runtime_training_invariants(state):
    identity = _bind_runtime_identity(state)
    quality = state.get("model_quality_snapshot") or {}
    snapshot = state.get("training_dataset")
    generation = state.get("training_generation")
    run_id = state.get("training_run_id")
    reasons = []
    if int(state.get("training_strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS:
        reasons.append("insufficient_pinned_strict_full_rows")
    if not quality.get("passed"):
        reasons.append("model_quality_gate_not_passed")
    if int(quality.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS:
        reasons.append("quality_snapshot_below_3000")
    if quality.get("training_run_id") != run_id:
        reasons.append("training_run_identity_mismatch")
    if (quality.get("solver_revision") != identity["solver_revision"]
            or quality.get("library_revision") != identity["library_revision"]):
        reasons.append("training_revision_identity_mismatch")
    if quality.get("quality_thresholds_sha256") != identity["quality_thresholds_sha256"]:
        reasons.append("quality_thresholds_identity_mismatch")
    if not snapshot or not os.path.isfile(snapshot):
        reasons.append("training_snapshot_missing")
    elif quality.get("dataset_sha256") != _sha256(snapshot):
        reasons.append("training_snapshot_fingerprint_mismatch")
    if not generation or not os.path.isdir(generation):
        reasons.append("training_generation_missing")
    else:
        report_path = os.path.join(generation, "train_report.json")
        try:
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            if (report.get("training_run_id") != run_id
                    or report.get("dataset_sha256") != quality.get("dataset_sha256")
                    or int(report.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS):
                reasons.append("training_generation_identity_mismatch")
        except Exception:
            reasons.append("training_generation_report_unavailable")
    if reasons:
        raise RuntimeError("AL training invariant failed: " + ";".join(reasons))


def _required_probe_targets(result):
    targets = [T_TARGETS[0], T_TARGETS[1], T_TARGETS[3]]
    if _finite(result.get("N2_side")) and float(result["N2_side"]) > 0:
        targets.insert(2, T_TARGETS[2])
    return targets


def _physical_llt(result):
    value = float(result["Llt"])
    return value if int(float(result["full_model"])) == 1 else 2.0 * value


def _result_passes_spec(result):
    """Apply physical FEA limits, independently from model agreement."""
    try:
        llt = _physical_llt(result)
        if not _finite(llt) or not _finite(result.get("B_max_core")):
            return False
        if not all(
                _finite(result.get(key))
                and 0 <= float(result[key]) <= SPEC["convergence_max_pct"]
                for key in ("conv_error_pct_matrix", "conv_error_pct_loss")):
            return False
        if not all(
                _finite(result.get(key)) and float(result[key]) >= 0
                for key in LOSS_COMPONENTS):
            return False
        if abs(llt - SPEC["Llt_target_uH"]) > SPEC["Llt_tol_uH"]:
            return False
        if not 0 <= float(result["B_max_core"]) <= SPEC["B_limit_T"]:
            return False
        for target in _required_probe_targets(result):
            actual_column = ACTUAL_TEMPERATURES[target]
            if not _finite(result.get(actual_column)):
                return False
            if float(result[actual_column]) > SPEC["T_limit_C"]:
                return False
        for column in (
                "cc_w2c_space_x", "cc_w2c_space_y",
                "w2c_w1c_space_x", "w2c_w1c_space_y",
                "w1c_w2s_gap_x_actual", "w1s_cs_space_x",
                "cs_w1s_space_y", "h_gap2"):
            if (not _finite(result.get(column))
                    or float(result[column]) < 40.0):
                return False
        if _finite(result.get("N1_side")) and float(result["N1_side"]) > 0:
            for column in ("w2s_w1s_space_x", "w1s_w2s_space_y"):
                if (not _finite(result.get(column))
                        or float(result[column]) < 40.0):
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _current_solver_revision():
    if PINNED_SOLVER_REVISION is not None:
        revision = str(PINNED_SOLVER_REVISION).strip().lower()
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise RuntimeError(f"invalid pinned solver git revision: {revision!r}")
        return revision
    repo_root = os.path.abspath(os.path.join(HERE, ".."))
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--",
         *SOLVER_REVISION_PATHS],
        cwd=repo_root, text=True).strip()
    if dirty:
        raise RuntimeError(
            "solver revision is not vetted: tracked solver inputs differ from HEAD\n" + dirty)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip().lower()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise RuntimeError(f"invalid solver git revision: {revision!r}")
    return revision


def _current_library_revision():
    if PINNED_LIBRARY_REVISION is not None:
        revision = str(PINNED_LIBRARY_REVISION).strip().lower()
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise RuntimeError(f"invalid pinned library git revision: {revision!r}")
        return revision
    library_root = os.environ.get("MFT_PYAEDT_LIBRARY_ROOT", "").strip()
    if not library_root:
        library_root = os.path.join(HERE, "..", "..", "pyaedt_library")
    library_root = os.path.abspath(library_root)
    if not os.path.exists(os.path.join(library_root, ".git")):
        raise RuntimeError(f"pyaedt_library git checkout is unavailable: {library_root}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
        cwd=library_root, text=True).strip()
    if dirty:
        raise RuntimeError(
            "pyaedt_library revision is not vetted: src differs from HEAD\n" + dirty)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=library_root, text=True).strip().lower()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise RuntimeError(f"invalid pyaedt_library git revision: {revision!r}")
    return revision


def _new_task_record(
        task_id, name=None, workdir=None, solver_revision=None,
        library_revision=None):
    return {
        "original_id": task_id,
        "retry_id": None,
        "active_id": task_id,
        "attempt": 0,
        "outcome": "pending" if task_id else "submission_unknown",
        "last_status": None,
        "result": None,
        "name": name,
        "workdir": workdir,
        "solver_git_revision": solver_revision,
        "pyaedt_library_git_revision": library_revision,
    }


def _ensure_task_records(st):
    """Migrate old task_map-only states without discarding task identity."""
    task_map = st.setdefault("task_map", {})
    records = st.setdefault("task_records", {})
    for idx, task_id in list(task_map.items()):
        key = str(idx)
        if key not in records:
            records[key] = _new_task_record(
                task_id,
                solver_revision=st.get("solver_git_revision"),
                library_revision=st.get("pyaedt_library_git_revision"))
        record = records[key]
        if record.get("solver_git_revision") is None and st.get("solver_git_revision"):
            record["solver_git_revision"] = st["solver_git_revision"]
        if (record.get("pyaedt_library_git_revision") is None
                and st.get("pyaedt_library_git_revision")):
            record["pyaedt_library_git_revision"] = st["pyaedt_library_git_revision"]
        if record.get("active_id") is None and task_id is not None:
            record["active_id"] = task_id
        task_map[key] = record.get("active_id")
    return records


def _stage_parquet(frame, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=os.path.dirname(target))
    os.close(fd)
    try:
        frame.to_parquet(staged, index=False)
        return staged
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise


def _atomic_write_parquet(frame, target):
    staged = _stage_parquet(frame, target)
    try:
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _atomic_write_csv(frame, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=os.path.dirname(target))
    os.close(fd)
    try:
        frame.to_csv(staged, index=False)
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _atomic_write_json(value, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp",
        dir=os.path.dirname(target),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, default=str)
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _atomic_save_npy(array, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array)
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _deduplicate_dataset(frame):
    """Deduplicate replayed AL rows while retaining legacy unidentified rows."""
    if frame is None or frame.empty:
        return frame
    identities = pd.Series(pd.NA, index=frame.index, dtype="object")
    if {"project_name", "saved_at"}.issubset(frame.columns):
        primary = (
            frame["project_name"].notna()
            & frame["saved_at"].notna()
            & frame["project_name"].astype(str).str.strip().ne("")
            & frame["saved_at"].astype(str).str.strip().ne("")
        )
        identities.loc[primary] = (
            "project:"
            + frame.loc[primary, "project_name"].astype(str)
            + "|"
            + frame.loc[primary, "saved_at"].astype(str)
        )
    if {"source", "task_id"}.issubset(frame.columns):
        fallback = identities.isna() & frame["source"].astype(str).str.startswith("al_round_") \
            & frame["task_id"].notna()
        identities.loc[fallback] = (
            "task:"
            + frame.loc[fallback, "source"].astype(str)
            + "|"
            + frame.loc[fallback, "task_id"].astype(str)
        )
    identified = identities.notna()
    keep = ~identified | ~identities.duplicated(keep="last")
    return frame.loc[keep].reset_index(drop=True)


def _unique_rows(array):
    if len(array) < 2:
        return array
    _, indices = np.unique(array, axis=0, return_index=True)
    return array[np.sort(indices)]


def _unit_vector_matrix(array, n_var, label, allow_incompatible=False):
    """Validate persisted Sobol vectors before distance or vstack operations."""
    matrix = np.asarray(array)
    valid = (
        matrix.ndim == 2
        and matrix.shape[1] == int(n_var)
        and np.issubdtype(matrix.dtype, np.number)
        and np.isfinite(matrix).all()
    )
    if valid:
        return matrix.astype(float, copy=False)
    message = (
        f"{label} is incompatible with the current Sobol schema: "
        f"shape={matrix.shape}, expected=(*, {n_var})"
    )
    if allow_incompatible:
        print(f"[al] {message}; ignoring legacy vectors")
        return None
    raise RuntimeError(message)


def _merge_source_ranks(existing, new_rows):
    keys = ["project_name", "saved_at"]
    columns = keys + [SOURCE_RANK_COLUMN]
    if not all(key in new_rows.columns for key in keys):
        raise RuntimeError("AL rows are missing source-rank identity columns")
    incoming = new_rows[keys].copy()
    incoming[SOURCE_RANK_COLUMN] = AL_SOURCE_RANK
    frames = [incoming]
    if existing is not None:
        missing = [column for column in columns if column not in existing.columns]
        if missing:
            raise RuntimeError(f"source rank sidecar schema is invalid; missing {missing}")
        sidecar = existing[columns].copy()
        complete = (
            sidecar["project_name"].notna()
            & sidecar["saved_at"].notna()
            & sidecar["project_name"].astype(str).str.strip().ne("")
            & sidecar["saved_at"].astype(str).str.strip().ne("")
        )
        numeric_rank = pd.to_numeric(sidecar[SOURCE_RANK_COLUMN], errors="coerce")
        if (not complete.all() or not np.isfinite(numeric_rank).all()
                or (numeric_rank < 0).any()
                or sidecar.duplicated(keys).any()):
            raise RuntimeError("source rank sidecar contains invalid or duplicate rows")
        sidecar[SOURCE_RANK_COLUMN] = numeric_rank
        frames.insert(0, sidecar)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    complete = (
        merged["project_name"].notna()
        & merged["saved_at"].notna()
        & merged["project_name"].astype(str).str.strip().ne("")
        & merged["saved_at"].astype(str).str.strip().ne("")
    )
    merged = merged.loc[complete].copy()
    merged[SOURCE_RANK_COLUMN] = pd.to_numeric(
        merged[SOURCE_RANK_COLUMN], errors="coerce").fillna(0)
    merged = merged.sort_values(SOURCE_RANK_COLUMN, kind="stable")
    return merged.drop_duplicates(keys, keep="last").reset_index(drop=True)


def _validated_existing_source_ranks(existing, existing_ranks, recoverable_rows=None):
    """Validate ranks, repairing only AL identities from an interrupted install."""
    keys = ["project_name", "saved_at"]
    empty_incoming = pd.DataFrame(columns=keys)
    validated = _merge_source_ranks(existing_ranks, empty_incoming)
    if existing is None or existing.empty:
        return validated
    if not all(key in existing.columns for key in keys):
        raise RuntimeError("existing dataset is missing source-rank identity columns")
    complete = (
        existing["project_name"].notna()
        & existing["saved_at"].notna()
        & existing["project_name"].astype(str).str.strip().ne("")
        & existing["saved_at"].astype(str).str.strip().ne("")
    )
    identities = existing.loc[complete, keys].drop_duplicates()
    coverage = identities.merge(validated[keys], on=keys, how="left", indicator=True)
    missing = coverage.loc[coverage["_merge"].ne("both"), keys]
    if len(missing):
        if recoverable_rows is None or not all(
                key in recoverable_rows.columns for key in keys):
            raise RuntimeError(
                "source rank sidecar does not cover every existing dataset identity")
        recoverable_complete = (
            recoverable_rows["project_name"].notna()
            & recoverable_rows["saved_at"].notna()
            & recoverable_rows["project_name"].astype(str).str.strip().ne("")
            & recoverable_rows["saved_at"].astype(str).str.strip().ne("")
        )
        recoverable = recoverable_rows.loc[recoverable_complete, keys].drop_duplicates()
        repairable = missing.merge(recoverable, on=keys, how="left", indicator=True)
        if not repairable["_merge"].eq("both").all():
            raise RuntimeError(
                "source rank sidecar does not cover every existing dataset identity")
        from campaign import collect_wave as collector
        collector._matching_replay_rows(
            existing, recoverable_rows, missing, keys)
        repaired = missing.copy()
        repaired[SOURCE_RANK_COLUMN] = AL_SOURCE_RANK
        validated = pd.concat([validated, repaired], ignore_index=True, sort=False)
    return validated


def _merge_ranked_dataset(existing, new_rows, existing_ranks):
    """Merge AL rows using the collector's source-rank replacement contract."""
    from campaign import collect_wave as collector

    keys = ["project_name", "saved_at"]
    incoming = new_rows.copy()
    incoming[SOURCE_RANK_COLUMN] = AL_SOURCE_RANK
    validated_ranks = _validated_existing_source_ranks(
        existing, existing_ranks, recoverable_rows=new_rows)
    ranked_existing = collector._attach_source_ranks(
        existing, validated_ranks, keys)
    accepted = collector.select_new_unique_rows(
        incoming, ranked_existing, keys)
    retained = collector._drop_replaced_key_rows(
        ranked_existing, accepted, keys)
    ranked = accepted if retained is None else pd.concat(
        [retained, accepted], ignore_index=True, sort=False)
    source_ranks = collector._source_rank_rows(ranked, keys)
    dataset = ranked.drop(columns=[SOURCE_RANK_COLUMN], errors="ignore")
    return _deduplicate_dataset(dataset), source_ranks


def _install_dataset_transaction(dataset, source_ranks, source_rank_path):
    """Serialize both artifacts before installing master first, then rank."""
    staged_dataset = _stage_parquet(dataset, DATASET)
    staged_ranks = None
    try:
        staged_ranks = _stage_parquet(source_ranks, source_rank_path)
        # Master first is recovery-safe: an interrupted write leaves the AL row
        # at its old lower rank, so replay can still promote it. Rank first could
        # label the old payload as AL authority and make replay reject the AL row.
        os.replace(staged_dataset, DATASET)
        staged_dataset = None
        os.replace(staged_ranks, source_rank_path)
        staged_ranks = None
    finally:
        for staged in (staged_ranks, staged_dataset):
            if staged and os.path.exists(staged):
                os.remove(staged)


def _verification_counts(records):
    total = len(records)
    valid = sum(record.get("outcome") == "valid" for record in records.values())
    exhausted = sum(record.get("outcome") == "exhausted" for record in records.values())
    return {
        "total": total,
        "valid": valid,
        "exhausted": exhausted,
        "pending": total - valid - exhausted,
        "coverage": (valid / total) if total else 0.0,
    }


def load_state():
    if os.path.isfile(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"round": 1, "stage": "TRAIN", "q_mult": 1.0, "task_map": {}, "history": []}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, STATE_PATH)


def run(cmd, **kw):
    print(f"[al] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


def stage_train(st):
    from quality_contract import annotate_validity

    _bind_runtime_identity(st)
    rnd = st["round"]
    rdir = os.path.join(_active_al_root(), f"round_{rnd:02d}")
    os.makedirs(rdir, exist_ok=True)
    audited = annotate_validity(
        pd.read_parquet(DATASET),
        expected_solver_revision=PINNED_SOLVER_REVISION,
        expected_library_revision=PINNED_LIBRARY_REVISION,
    )
    strict = audited.loc[audited["_strict_valid_full"]].drop(
        columns=[
            "_strict_valid_em", "_strict_valid_thermal",
            "_strict_valid_full", "_strict_invalid_reasons",
        ],
        errors="ignore",
    )
    if len(strict) < MIN_STRICT_FULL_ROWS:
        raise RuntimeError(
            f"AL requires at least {MIN_STRICT_FULL_ROWS} pinned strict-full rows; "
            f"found {len(strict)}"
        )
    model_dataset = os.path.join(rdir, "strict_training_snapshot.parquet")
    _atomic_write_parquet(strict, model_dataset)
    from training.train_models import (
        discard_inactive_generation, promote_generation,
        registry_pointer_token,
    )
    from quality_contract import DEFAULT_PROFILE_PATH

    profile_path = os.path.abspath(DEFAULT_PROFILE_PATH)
    from quality_contract import load_profile

    profile_sha256 = hashlib.sha256(
        json.dumps(
            load_profile(profile_path), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    candidate_result_path = os.path.join(rdir, "candidate_generation.json")
    expected_pointer = registry_pointer_token(REGISTRY)
    candidate_generation = None
    promoted = False
    try:
        run([
            PY, os.path.join("training", "checkpoint_train.py"),
            "--dataset", model_dataset,
            "--curve-csv", os.path.join(
                RUNTIME_ROOT, "training", "learning_curve.csv"
            ),
            "--profile", profile_path,
        ])
        run([
            PY, os.path.join("training", "train_models.py"),
            "--dataset", model_dataset, "--registry", REGISTRY,
            "--profile", profile_path,
            "--result-json", candidate_result_path,
        ])
        with open(candidate_result_path, encoding="utf-8") as handle:
            candidate = json.load(handle)
        candidate_generation = candidate["generation"]
        from training.model_quality_gate import evaluate_generation

        thresholds_path = QUALITY_THRESHOLDS_PATH
        with open(thresholds_path, encoding="utf-8") as handle:
            thresholds = json.load(handle)
        quality_snapshot = evaluate_generation(
            REGISTRY, candidate_generation, model_dataset, thresholds
        )
        quality_snapshot.update({
            "solver_revision": PINNED_SOLVER_REVISION,
            "library_revision": PINNED_LIBRARY_REVISION,
            "quality_thresholds_sha256": _sha256(QUALITY_THRESHOLDS_PATH),
        })
        quality_snapshot["evaluated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        _atomic_write_json(quality_snapshot, QUALITY_STATUS)
        if not quality_snapshot.get("passed"):
            raise RuntimeError(
                "surrogate quality gate failed: "
                + "; ".join(quality_snapshot.get("reasons", [])[:10])
            )
        if (int(quality_snapshot.get("strict_full_rows") or 0)
                < MIN_STRICT_FULL_ROWS):
            raise RuntimeError("surrogate quality gate is below 3000 strict rows")
        pointer = promote_generation(
            REGISTRY,
            candidate_generation,
            quality_snapshot,
            dataset=model_dataset,
            profile_sha256=profile_sha256,
            thresholds_sha256=hashlib.sha256(
                json.dumps(
                    thresholds, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            expected_pointer=expected_pointer,
        )
        promoted = True
        generation_dir = os.path.abspath(os.path.join(REGISTRY, pointer["generation"]))
        quality_snapshot_path = os.path.join(rdir, "model_quality_snapshot.json")
        _atomic_write_json(quality_snapshot, quality_snapshot_path)
    except Exception:
        if candidate_generation and not promoted:
            try:
                discard_inactive_generation(REGISTRY, candidate_generation)
            except Exception as cleanup_error:
                print(f"[al] inactive candidate cleanup warning: {cleanup_error}")
        raise
    st["training_dataset"] = model_dataset
    st["training_run_id"] = pointer["training_run_id"]
    st["training_generation"] = generation_dir
    st["model_quality_snapshot"] = quality_snapshot
    st["model_quality_snapshot_path"] = quality_snapshot_path
    st["training_strict_full_rows"] = int(len(strict))
    st["stage"] = "OPTIMIZE"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_training_invariants(st):
    """Fail closed if the pinned accepted training evidence changes mid-round."""
    _assert_runtime_training_invariants(st)
    required = (
        "training_dataset", "training_run_id", "training_generation",
        "model_quality_snapshot_path", "training_strict_full_rows",
    )
    missing = [key for key in required if st.get(key) in (None, "")]
    if missing:
        raise RuntimeError("AL training evidence is missing: " + ", ".join(missing))
    dataset = os.path.abspath(st["training_dataset"])
    if not os.path.isfile(dataset):
        raise RuntimeError("AL pinned training snapshot is missing")
    from training.train_models import load_generation

    record = load_generation(
        REGISTRY, st["training_generation"], require_accepted=True
    )
    report = record["report"]
    accepted = record["quality"]
    quality_path = os.path.abspath(st["model_quality_snapshot_path"])
    with open(quality_path, encoding="utf-8") as handle:
        quality_snapshot = json.load(handle)
    checks = {
        "training_run_id": st["training_run_id"],
        "dataset_sha256": _file_sha256(dataset),
        "generation": record["generation_relative"],
        "generation_report_sha256": record["generation_report_sha256"],
        "profile_sha256": report.get("profile_sha256"),
        "thresholds_sha256": accepted.get("thresholds_sha256"),
    }
    if os.path.normcase(os.path.abspath(st["training_generation"])) != os.path.normcase(
        record["generation"]
    ):
        raise RuntimeError("AL pinned generation path changed")
    if int(st["training_strict_full_rows"]) != int(report.get("strict_full_rows", -1)):
        raise RuntimeError("AL pinned strict row count changed")
    for key, expected in checks.items():
        if report.get(key) not in (None, expected) and key in (
            "training_run_id", "dataset_sha256", "profile_sha256"
        ):
            raise RuntimeError(f"AL generation {key} changed")
        if quality_snapshot.get(key) != expected:
            raise RuntimeError(f"AL quality snapshot {key} changed")
        if accepted.get(key) not in (None, expected):
            raise RuntimeError(f"AL accepted gate {key} changed")
    if quality_snapshot.get("passed") is not True or accepted.get("passed") is not True:
        raise RuntimeError("AL pinned generation is not quality accepted")


def stage_optimize(st):
    _assert_training_invariants(st)
    rnd = st["round"]
    spec_path = os.path.join(_active_al_root(), f"round_{rnd:02d}_spec.json")
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    # q 적응 배율 반영
    spec = {"q_sigma": 1.28 * st.get("q_mult", 1.0)}
    json.dump(spec, open(spec_path, "w"))
    run([PY, os.path.join("optimization", "run_nsga2.py"),
         "--restarts", "16", "--round", str(rnd), "--spec", spec_path,
         "--dataset", st["training_dataset"], "--registry", REGISTRY,
         "--registry-generation", st["training_generation"],
         "--quality-status", st["model_quality_snapshot_path"],
         "--output-root", _active_al_root()])
    st["stage"] = "SELECT"


def stage_select(st):
    _assert_training_invariants(st)
    rnd = st["round"]
    rdir = os.path.join(_active_al_root(), f"round_{rnd:02d}")
    X = np.load(os.path.join(rdir, "pareto_X.npy"))
    from module.input_parameter_260706 import _SOBOL_DIMS

    X = _unit_vector_matrix(X, len(_SOBOL_DIMS), "pareto_X")
    F = np.load(os.path.join(rdir, "pareto_F.npy"))
    front = pd.read_csv(os.path.join(rdir, "pareto_front.csv"))

    sig_cols = [c for c in front.columns if c.startswith("sigma_")]
    sig_norm = front[sig_cols].div(front[sig_cols].mean() + 1e-12).sum(axis=1).to_numpy()
    G = np.zeros((len(X), 1))  # Pareto는 feasible만 오므로 마진은 예측치로 재구성
    margin = (SPEC["Llt_tol_uH"] - (front["pred_Llt_phys"] - SPEC["Llt_target_uH"]).abs()).to_numpy()
    t_margin = np.min([(SPEC["T_limit_C"] - front[f"pred_{t}"]).to_numpy()
                       for t in T_TARGETS if f"pred_{t}" in front.columns] or [np.full(len(X), 99.0)], axis=0)
    G = -np.column_stack([margin, t_margin])  # select()는 -G를 마진으로 봄

    verified_X = None
    vx_path = os.path.join(_active_al_root(), "verified_X.npy")
    if os.path.isfile(vx_path):
        verified_X = _unit_vector_matrix(
            np.load(vx_path), len(_SOBOL_DIMS), "verified_X",
            allow_incompatible=True,
        )

    from select_candidates import select
    picked = select(X, F, G, sig_norm, verified_X=verified_X)
    if len(picked) != min(SPEC["K"], len(X)):
        raise RuntimeError(
            f"candidate selection returned {len(picked)} rows; "
            f"expected {min(SPEC['K'], len(X))}"
        )
    np.save(os.path.join(rdir, "selected_idx.npy"), np.array(picked))
    print(f"[al] round {rnd}: {len(picked)} candidates selected")
    st["stage"] = "SUBMIT"


def stage_submit(st):
    _assert_training_invariants(st)
    if not EXECUTE_SUBMISSIONS:
        raise RuntimeError(
            "standard FEA submission is disabled; rerun with --execute after reviewing artifacts"
        )
    _require_runtime_deployment()
    rnd = st["round"]
    rdir = os.path.join(_active_al_root(), f"round_{rnd:02d}")
    picked = np.load(os.path.join(rdir, "selected_idx.npy")).tolist()
    front = pd.read_csv(os.path.join(rdir, "pareto_front.csv"))

    from module.input_parameter_260706 import KEYS  # noqa
    import scheduler_client as sc
    with open(os.path.join(HERE, "verify", "profiles", "standard.json"), encoding="utf-8") as profile_file:
        profile = json.load(profile_file)

    if st.get("task_round") != rnd:
        st["task_map"] = {}
        st["task_records"] = {}
        st["task_round"] = rnd
        st["solver_git_revision"] = _current_solver_revision()
        st["pyaedt_library_git_revision"] = _current_library_revision()
        save_state(st)
    solver_revision = st.get("solver_git_revision")
    if not isinstance(solver_revision, str) or len(solver_revision) != 40:
        raise RuntimeError("AL round has no pinned full solver git revision")
    library_revision = st.get("pyaedt_library_git_revision")
    if not isinstance(library_revision, str) or len(library_revision) != 40:
        raise RuntimeError("AL round has no pinned full pyaedt_library git revision")
    task_map = st.setdefault("task_map", {})
    records = st.setdefault("task_records", {})
    unknown = []
    for j, i in enumerate(picked):
        idx_str = str(i)
        if idx_str in records:
            if records[idx_str].get("outcome") == "submission_unknown":
                unknown.append(idx_str)
            continue
        row = front.iloc[i]
        params = {k: (row[k] if not isinstance(row[k], np.generic) else row[k].item())
                  for k in KEYS if k in front.columns and pd.notna(row[k])}
        if set(params) != set(KEYS):
            missing = sorted(set(KEYS) - set(params))
            raise RuntimeError(f"standard candidate is missing required inputs: {missing}")
        submitted_params = sc.effective_verification_params(params, profile)
        if set(submitted_params) != set(KEYS):
            raise RuntimeError("standard profile produced an incomplete parameter payload")
        name = f"mft-al-r{rnd:02d}-c{j:02d}"
        workdir = f"mft_al_r{rnd:02d}_c{j:02d}"
        tid = sc.submit_verification(
            name, workdir, params, profile,
            mem_mb=profile.get("mem_mb", 32768), cpus=profile.get("cpus", 4),
            solver_revision=solver_revision, library_revision=library_revision)
        task_map[idx_str] = tid
        records[idx_str] = _new_task_record(
            tid, name=name, workdir=workdir, solver_revision=solver_revision,
            library_revision=library_revision)
        records[idx_str]["submitted_params"] = submitted_params
        if tid is None:
            unknown.append(idx_str)
        # Persist every returned identity. A driver crash cannot silently submit
        # the same candidate again on the next invocation.
        save_state(st)
        time.sleep(0.5)
    if unknown:
        raise RuntimeError(
            f"scheduler accepted or rejected submissions without recoverable task IDs: {unknown}"
        )
    st["stage"] = "WAIT"


def stage_wait(st):
    _assert_training_invariants(st)
    import scheduler_client as sc
    records = _ensure_task_records(st)
    solver_revision = st.get("solver_git_revision")
    if not isinstance(solver_revision, str) or len(solver_revision) != 40:
        raise RuntimeError("cannot verify tasks without the round's pinned solver revision")
    library_revision = st.get("pyaedt_library_git_revision")
    if not isinstance(library_revision, str) or len(library_revision) != 40:
        raise RuntimeError("cannot verify tasks without the pinned pyaedt_library revision")
    unresolved = {
        idx: record for idx, record in records.items()
        if record.get("outcome") not in ("valid", "exhausted")
    }
    unknown_submissions = [
        idx for idx, record in unresolved.items()
        if (record.get("active_id") is None
            or record.get("outcome") in ("submission_unknown", "retry_submission_unknown"))
    ]
    if unknown_submissions:
        st["verification_counts"] = dict(
            round=st["round"], **_verification_counts(records))
        st["stage"] = "WAIT"
        save_state(st)
        raise RuntimeError(
            f"cannot safely resubmit candidates with unknown task identity: {unknown_submissions}"
        )

    tids = [record["active_id"] for record in unresolved.values()]
    retry_only = bool(tids) and all(record.get("attempt", 0) >= 1 for record in unresolved.values())
    status = sc.wait_all(
        tids, poll_s=180, timeout_s=(5 if retry_only else 6) * 3600)

    # A terminal task with no valid data is retried once at 64 GB. Active or
    # unknown states remain in WAIT and never consume a duplicate allocation.
    rnd = st["round"]
    front = pd.read_csv(os.path.join(_active_al_root(), f"round_{rnd:02d}", "pareto_front.csv"))
    from module.input_parameter_260706 import KEYS
    profile_path = os.path.join(HERE, "verify", "profiles", "standard.json")
    with open(profile_path, encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    retried = 0
    fetch_errors = []
    submission_unknown = []
    for idx_str, record in unresolved.items():
        tid = record["active_id"]
        task_status = status.get(tid)
        record["last_status"] = task_status
        if task_status not in TERMINAL_TASK_STATUSES:
            continue

        try:
            expected_revision = record.get("solver_git_revision") or solver_revision
            expected_library = (
                record.get("pyaedt_library_git_revision") or library_revision)
            fetched = sc.fetch_result(
                tid, expected_revision=expected_revision,
                expected_library_revision=expected_library)
        except sc.ResultFetchError as exc:
            record["outcome"] = "fetch_error"
            record["fetch_error"] = str(exc)
            fetch_errors.append((idx_str, str(exc)))
            save_state(st)
            continue

        record["last_result_state"] = fetched.state
        row = front.iloc[int(idx_str)]
        params = {
            key: (row[key].item() if hasattr(row[key], "item") else row[key])
            for key in KEYS if key in front.columns and pd.notna(row[key])
        }
        submitted_params = record.get("submitted_params")
        if not isinstance(submitted_params, dict):
            submitted_params = sc.effective_verification_params(params, profile)
            record["submitted_params"] = submitted_params
        result_matches = (
            fetched.state == sc.RESULT_VALID
            and sc.result_matches_params(
                fetched.result, submitted_params, required_keys=KEYS
            )
        )
        if result_matches:
            record["result"] = fetched.result
            record["outcome"] = "valid"
            record.pop("fetch_error", None)
            save_state(st)
            continue
        if fetched.state == sc.RESULT_VALID:
            record["last_result_state"] = "candidate_result_identity_mismatch"

        if int(record.get("attempt", 0)) >= 1:
            record["outcome"] = "exhausted"
            save_state(st)
            continue

        retry_name = f"mft-al-r{rnd:02d}-retry-{idx_str}"
        retry_workdir = f"mft_al_r{rnd:02d}_rt{idx_str}"
        _require_runtime_deployment()
        new_tid = sc.submit_verification(
            retry_name, retry_workdir, params, profile,
            mem_mb=65536, cpus=profile.get("cpus", 4),
            solver_revision=solver_revision, library_revision=library_revision)
        if new_tid is None:
            record["outcome"] = "retry_submission_unknown"
            submission_unknown.append(idx_str)
            save_state(st)
            continue
        record.update({
            "retry_id": new_tid,
            "active_id": new_tid,
            "attempt": 1,
            "outcome": "pending",
            "last_status": None,
            "solver_git_revision": solver_revision,
            "pyaedt_library_git_revision": library_revision,
        })
        st["task_map"][idx_str] = new_tid
        retried += 1
        # Persist the original and retry IDs before waiting on the retry.
        save_state(st)

    if retried:
        print(f"[al] {retried} candidates retried at 64GB")
    if submission_unknown:
        st["verification_counts"] = dict(
            round=st["round"], **_verification_counts(records))
        st["stage"] = "WAIT"
        save_state(st)
        raise RuntimeError(
            f"retry submissions have unknown task identity: {submission_unknown}"
        )
    if fetch_errors:
        st["verification_counts"] = dict(
            round=st["round"], **_verification_counts(records))
        st["stage"] = "WAIT"
        save_state(st)
        raise RuntimeError(
            f"scheduler stdout remained unavailable; no tasks were resubmitted: {fetch_errors}"
        )

    counts = _verification_counts(records)
    st["verification_counts"] = dict(round=st["round"], **counts)
    n_total = counts["total"]
    n_done = counts["valid"]
    n_exhausted = counts["exhausted"]
    n_pending = counts["pending"]
    print(f"[al] wait status: valid={n_done}/{n_total}, exhausted={n_exhausted}, pending={n_pending}")
    if n_pending:
        st["stage"] = "WAIT"
        return
    if n_total and n_done < 0.7 * n_total:
        print("[al] WARNING: <70% valid completion - 실패 태스크 로그 점검 필요")
    st["stage"] = "INGEST"


def stage_ingest(st):
    _assert_training_invariants(st)
    rnd = st["round"]
    rdir = os.path.join(_active_al_root(), f"round_{rnd:02d}")
    front = pd.read_csv(os.path.join(rdir, "pareto_front.csv"))
    import scheduler_client as sc
    with open(
        os.path.join(CODE_ROOT, "verify", "profiles", "standard.json"),
        encoding="utf-8",
    ) as profile_file:
        profile = json.load(profile_file)

    records = _ensure_task_records(st)
    solver_revision = st.get("solver_git_revision")
    if not isinstance(solver_revision, str) or len(solver_revision) != 40:
        raise RuntimeError("cannot ingest without the round's pinned solver revision")
    library_revision = st.get("pyaedt_library_git_revision")
    if not isinstance(library_revision, str) or len(library_revision) != 40:
        raise RuntimeError("cannot ingest without the pinned pyaedt_library revision")
    st["verification_counts"] = dict(
        round=rnd, **_verification_counts(records))
    rows, errs = [], []
    for idx_str, record in records.items():
        if record.get("outcome") == "exhausted":
            continue
        res = record.get("result")
        if res is None:
            tid = record.get("active_id")
            if tid is None:
                st["stage"] = "WAIT"
                return
            try:
                expected_revision = record.get("solver_git_revision") or solver_revision
                expected_library = (
                    record.get("pyaedt_library_git_revision") or library_revision)
                fetched = sc.fetch_result(
                    tid, expected_revision=expected_revision,
                    expected_library_revision=expected_library)
            except sc.ResultFetchError as exc:
                record["outcome"] = "fetch_error"
                record["fetch_error"] = str(exc)
                st["stage"] = "INGEST"
                save_state(st)
                raise RuntimeError(
                    f"cannot ingest task {tid}: scheduler stdout unavailable"
                ) from exc
            if fetched.state != sc.RESULT_VALID:
                record["last_result_state"] = fetched.state
                record["outcome"] = "pending"
                st["stage"] = "WAIT"
                save_state(st)
                return
            res = fetched.result
            record["result"] = res
            record["outcome"] = "valid"
            save_state(st)
        expected_revision = record.get("solver_git_revision") or solver_revision
        expected_library = (
            record.get("pyaedt_library_git_revision") or library_revision)
        if not sc.is_valid_result(
                res, expected_revision=expected_revision,
                expected_library_revision=expected_library):
            record["outcome"] = "pending"
            record["result"] = None
            st["stage"] = "WAIT"
            save_state(st)
            return
        from module.input_parameter_260706 import KEYS
        candidate_row = front.iloc[int(idx_str)]
        candidate_params = {
            key: (candidate_row[key].item()
                  if hasattr(candidate_row[key], "item") else candidate_row[key])
            for key in KEYS
            if key in front.columns and pd.notna(candidate_row[key])
        }
        submitted_params = record.get("submitted_params")
        if not isinstance(submitted_params, dict):
            submitted_params = sc.effective_verification_params(
                candidate_params, profile
            )
        if not sc.result_matches_params(
                res, submitted_params, required_keys=KEYS):
            record["outcome"] = "pending"
            record["last_result_state"] = "candidate_result_identity_mismatch"
            record["result"] = None
            st["stage"] = "WAIT"
            save_state(st)
            return
        tid = record.get("active_id")
        row_result = dict(res)
        row_result["source"] = f"al_round_{rnd}"
        row_result["sample_weight"] = 3.0
        row_result["task_id"] = tid
        rows.append(row_result)
        i = int(idx_str)
        pred = front.iloc[i]
        llt_fea = _physical_llt(res)
        llt_pred = float(pred["pred_Llt_phys"]) if _finite(pred.get("pred_Llt_phys")) else np.nan
        err = {
            "idx": i, "task_id": tid,
            "llt_pred": llt_pred, "llt_fea": llt_fea,
            "dllt_pct": (abs(llt_pred - llt_fea) / SPEC["Llt_target_uH"] * 100
                           if _finite(llt_pred) else np.nan),
            "B_fea": float(res["B_max_core"]),
            "spec_pass": int(_result_passes_spec(res)),
        }
        predicted_losses = [pred.get(f"pred_{key}", np.nan) for key in LOSS_COMPONENTS]
        fea_losses = [res.get(key, np.nan) for key in LOSS_COMPONENTS]
        loss_pred = sum(float(value) for value in predicted_losses) \
            if all(_finite(value) for value in predicted_losses) else np.nan
        loss_fea = sum(float(value) for value in fea_losses) \
            if all(_finite(value) for value in fea_losses) else np.nan
        err["loss_pred_W"] = loss_pred
        err["loss_fea_W"] = loss_fea
        err["dloss_pct"] = (
            abs(loss_pred - loss_fea) / max(abs(loss_fea), 1e-9) * 100
            if _finite(loss_pred) and _finite(loss_fea) else np.nan
        )
        expected_targets = _required_probe_targets(res)
        err["temperature_error_expected_count"] = len(expected_targets)
        for target in expected_targets:
            prediction = pred.get(f"pred_{target}", np.nan)
            err[f"d_{target}"] = (
                abs(float(prediction) - float(res[target]))
                if _finite(prediction) and _finite(res.get(target)) else np.nan
            )
            err[f"fea_{ACTUAL_TEMPERATURES[target]}"] = float(
                res[ACTUAL_TEMPERATURES[target]])
        err["temperature_error_complete"] = int(all(
            _finite(err.get(f"d_{target}")) for target in expected_targets))
        errs.append(err)

    if rows:
        new = pd.DataFrame(rows)
        # The campaign collector uses this exact lock. Re-read under the lock,
        # deduplicate replayed AL rows, then atomically replace the master file.
        os.makedirs(os.path.dirname(DATASET), exist_ok=True)
        with FileLock(DATASET + ".lock", timeout=120):
            old = pd.read_parquet(DATASET) if os.path.isfile(DATASET) else None
            source_rank_path = os.path.join(
                os.path.dirname(DATASET), "source_ranks.parquet")
            old_ranks = (pd.read_parquet(source_rank_path)
                         if os.path.isfile(source_rank_path) else None)
            allf, source_ranks = _merge_ranked_dataset(old, new, old_ranks)
            _install_dataset_transaction(allf, source_ranks, source_rank_path)

            # Verified X is also replay-safe. A crash between files is repaired
            # by re-entering INGEST without duplicating either artifact.
            X = np.load(os.path.join(rdir, "pareto_X.npy"))
            from module.input_parameter_260706 import _SOBOL_DIMS

            X = _unit_vector_matrix(X, len(_SOBOL_DIMS), "pareto_X")
            v_new = X[[e["idx"] for e in errs]]
            vx_path = os.path.join(_active_al_root(), "verified_X.npy")
            previous = None
            if os.path.isfile(vx_path):
                previous = _unit_vector_matrix(
                    np.load(vx_path), len(_SOBOL_DIMS), "verified_X",
                    allow_incompatible=True,
                )
            v_all = np.vstack([previous, v_new]) if previous is not None else v_new
            _atomic_save_npy(_unique_rows(v_all), vx_path)

    err_df = pd.DataFrame(errs)
    _atomic_write_csv(err_df, os.path.join(rdir, "verification_errors.csv"))
    st["last_errs"] = err_df.to_dict("list") if len(err_df) else {}
    counts = _verification_counts(records)
    counts["ingested"] = len(rows)
    st["verification_counts"] = dict(round=rnd, **counts)
    print(f"[al] ingested {len(rows)} verified rows")
    st["stage"] = "CHECK"


def stage_check(st):
    _assert_training_invariants(st)
    rnd = st["round"]
    errs = st.get("last_errs") or {}
    hist = {"round": rnd, "time": datetime.now().isoformat(timespec="seconds")}
    verification = st.get("verification_counts") or {}
    verification_total = int(verification.get("total") or 0)
    verification_valid = int(verification.get("valid") or 0)
    verification_exhausted = int(verification.get("exhausted") or 0)
    verification_pending = int(verification.get("pending") or 0)
    verification_ingested = int(verification.get("ingested") or 0)
    verification_coverage = (
        verification_valid / verification_total if verification_total else 0.0)
    verification_coverage_ok = bool(
        verification.get("round") == rnd
        and verification_total >= SPEC["K"]
        and verification_pending == 0
        and verification_coverage >= SPEC["verification_min_coverage"])
    hist.update({
        "verification_total": verification_total,
        "verification_valid": verification_valid,
        "verification_exhausted": verification_exhausted,
        "verification_ingested": verification_ingested,
        "verification_coverage": verification_coverage,
        "verification_coverage_ok": verification_coverage_ok,
    })

    ok_specs = 0
    agree = False
    if errs and errs.get("dllt_pct"):
        d = np.array(errs["dllt_pct"], dtype=float)
        verification_rows_complete = bool(
            len(d) == verification_valid == verification_ingested)
        hist["verification_rows_complete"] = verification_rows_complete
        d_finite = d[np.isfinite(d)]
        d_complete = len(d_finite) == len(d) and len(d) > 0
        hist["dllt_med_pct"] = float(np.median(d_finite)) if len(d_finite) else None
        hist["dllt_max_pct"] = float(np.max(d_finite)) if len(d_finite) else None
        t_cols = [k for k in errs if k.startswith("d_Tprobe")]
        dT = np.array([v for k in t_cols for v in errs[k]], dtype=float) if t_cols else np.array([])
        dT = dT[np.isfinite(dT)]
        expected_counts = np.array(
            errs.get("temperature_error_expected_count", []), dtype=float)
        expected_total = int(expected_counts.sum()) if (
            len(expected_counts) == len(d) and np.isfinite(expected_counts).all()) else 0
        coverage_flags = np.array(
            errs.get("temperature_error_complete", []), dtype=float)
        rows_complete = bool(
            len(coverage_flags) == len(d)
            and np.isfinite(coverage_flags).all()
            and (coverage_flags == 1).all())
        hist["temperature_error_count"] = int(len(dT))
        hist["temperature_error_expected_count"] = expected_total
        hist["temperature_error_coverage_complete"] = bool(
            rows_complete and expected_total > 0 and len(dT) == expected_total)
        hist["dT_med"] = float(np.median(dT)) if len(dT) else None
        hist["dT_max"] = float(np.max(dT)) if len(dT) else None

        dloss = np.array(errs.get("dloss_pct", []), dtype=float)
        dloss_finite = dloss[np.isfinite(dloss)]
        loss_complete = len(dloss) == len(d) and len(dloss_finite) == len(d)
        hist["loss_error_count"] = int(len(dloss_finite))
        hist["loss_error_expected_count"] = int(len(d))
        hist["loss_error_coverage_complete"] = bool(loss_complete)
        hist["dP_med_pct"] = (
            float(np.median(dloss_finite)) if len(dloss_finite) else None)
        hist["dP_max_pct"] = (
            float(np.max(dloss_finite)) if len(dloss_finite) else None)

        agree = (d_complete
                 and verification_rows_complete
                 and hist["dllt_med_pct"] <= SPEC["agree_llt_med_pct"]
                 and hist["dllt_max_pct"] <= SPEC["agree_llt_max_pct"]
                 and hist["temperature_error_coverage_complete"]
                 and hist["dT_med"] <= SPEC["agree_T_med_C"]
                 and hist["dT_max"] <= SPEC["agree_T_max_C"]
                 and loss_complete
                 and hist["dP_med_pct"] <= SPEC["agree_P_med_pct"]
                 and hist["dP_max_pct"] <= SPEC["agree_P_max_pct"])

        # 실측 스펙 통과 후보 수: 동일 후보가 Llt, 모든 적용 온도, B를
        # 동시에 통과해야 한 건으로 센다.
        llt_fea = np.array(errs["llt_fea"], dtype=float)
        band = np.abs(llt_fea - SPEC["Llt_target_uH"]) <= SPEC["Llt_tol_uH"]
        hist["fea_llt_pass"] = int(band.sum())
        spec_flags = np.array(errs.get("spec_pass", []), dtype=float)
        if len(spec_flags) == len(d) and np.isfinite(spec_flags).all():
            ok_specs = int((spec_flags == 1).sum())
        hist["fea_full_spec_pass"] = ok_specs

        # 예측통과/실측탈락 발생 시 q 조임
        miss = len(d) - ok_specs
        if miss > 0:
            st["q_mult"] = min(st.get("q_mult", 1.0) * 1.25, 3.0)
        elif hist["dllt_max_pct"] is not None and hist["dllt_max_pct"] < 0.5 * SPEC["agree_llt_max_pct"]:
            st["q_mult"] = max(st.get("q_mult", 1.0) * 0.9, 1.0)
        hist["q_mult"] = st["q_mult"]

    st.setdefault("history", []).append(hist)
    print(f"[al] round {rnd} check: {hist}")

    if agree and ok_specs >= 3 and verification_coverage_ok:
        if not st.get("post_convergence_retrain_done"):
            # The rows just ingested were not in the model that proposed this
            # front.  One mandatory retrain -> NSGA-II -> standard-FEA round
            # removes that one-generation lag before fine verification.
            st["post_convergence_retrain_done"] = True
            st["round"] = rnd + 1
            st["stage"] = "TRAIN"
            print("[al] standard agreement reached; mandatory final retrain scheduled")
        else:
            st["stage"] = "FINAL_SELECT"
            print("[al] standard loop converged; selecting minimum-volume fine candidates")
    elif rnd >= SPEC["max_rounds"]:
        # A compute budget cap is terminal, but it is not a verified design.
        st["stage"] = "HARD_CAP"
        print("[al] hard cap reached without convergence")
    else:
        st["round"] = rnd + 1
        st["stage"] = "TRAIN"


def stage_final_select(st):
    _assert_training_invariants(st)
    from verify.finalize import collect_standard_pass_candidates

    queue = collect_standard_pass_candidates(
        _active_al_root(), DATASET, limit=None,
        expected_solver_revision=(
            st.get("solver_git_revision") or PINNED_SOLVER_REVISION
        ),
        expected_library_revision=(
            st.get("pyaedt_library_git_revision") or PINNED_LIBRARY_REVISION
        ),
    )
    attempted = {
        item.get("candidate_digest")
        for item in st.get("fine_attempt_history", [])
        if isinstance(item, dict) and item.get("candidate_digest")
    }
    queue = [
        candidate for candidate in queue
        if candidate["candidate_digest"] not in attempted
    ]
    if not queue:
        raise RuntimeError(
            "no strict standard-FEA spec-pass candidate is available for fine verification"
        )
    st["fine_candidate_queue"] = queue
    st["fine_queue_cursor"] = 0
    st["fine_batch"] = 0
    st["final_candidates"] = queue[:3]
    st["fine_task_records"] = {}
    results_dir = os.path.join(OUTPUT_ROOT, "verify", "results")
    os.makedirs(results_dir, exist_ok=True)
    _atomic_write_csv(
        pd.DataFrame([
            {
                "rank": rank,
                "round": candidate["round"],
                "index": candidate["index"],
                "task_id": candidate["task_id"],
                "volume_L": candidate["volume_L"],
                "total_loss_W": candidate["total_loss_W"],
                "candidate_digest": candidate["candidate_digest"],
            }
            for rank, candidate in enumerate(queue)
        ]),
        os.path.join(results_dir, "fine_candidate_queue.csv"),
    )
    st["stage"] = "FINE_SUBMIT"


def stage_fine_submit(st):
    _assert_training_invariants(st)
    if not EXECUTE_SUBMISSIONS:
        raise RuntimeError(
            "fine FEA submission is disabled; inspect fine_candidate_queue.csv "
            "and rerun with --execute"
        )
    _require_runtime_deployment()
    import scheduler_client as sc
    from module.input_parameter_260706 import KEYS

    profile_path = os.path.join(HERE, "verify", "profiles", "fine.json")
    with open(profile_path, encoding="utf-8") as handle:
        profile = json.load(handle)
    solver_revision = _current_solver_revision()
    library_revision = _current_library_revision()
    st["fine_solver_git_revision"] = solver_revision
    st["fine_pyaedt_library_git_revision"] = library_revision
    records = st.setdefault("fine_task_records", {})
    unknown = []
    for rank, candidate in enumerate(st["final_candidates"]):
        key = str(rank)
        if key in records:
            if records[key].get("active_id") is None:
                unknown.append(key)
            continue
        batch = int(st.get("fine_batch", 0))
        fine_params = sc.effective_verification_params(candidate["params"], profile)
        if set(fine_params) != set(KEYS):
            missing = sorted(set(KEYS) - set(fine_params))
            raise RuntimeError(f"fine candidate is missing required inputs: {missing}")
        candidate["fine_params"] = fine_params
        name = f"mft-final-fine-r{st['round']:02d}-b{batch:02d}-c{rank:02d}"
        workdir = f"mft_final_fine_r{st['round']:02d}_b{batch:02d}_c{rank:02d}"
        task_id = sc.submit_verification(
            name,
            workdir,
            candidate["params"],
            profile,
            mem_mb=profile["mem_mb"],
            cpus=profile["cpus"],
            solver_revision=solver_revision,
            library_revision=library_revision,
        )
        records[key] = _new_task_record(
            task_id,
            name=name,
            workdir=workdir,
            solver_revision=solver_revision,
            library_revision=library_revision,
        )
        if task_id is None:
            unknown.append(key)
        save_state(st)
    if unknown:
        raise RuntimeError(f"fine submissions have unknown identities: {unknown}")
    st["stage"] = "FINE_WAIT"


def stage_fine_wait(st):
    _assert_training_invariants(st)
    import scheduler_client as sc
    from module.input_parameter_260706 import KEYS

    records = st.get("fine_task_records") or {}
    if len(records) != len(st.get("final_candidates") or []):
        raise RuntimeError("fine task inventory is incomplete")
    pending_records = {
        key: record for key, record in records.items()
        if record.get("outcome") not in ("valid", "unverified")
    }
    task_ids = [record.get("active_id") for record in pending_records.values()]
    if any(task_id is None for task_id in task_ids):
        raise RuntimeError("fine task identity is unknown; refusing duplicate submission")
    status = sc.wait_all(task_ids, poll_s=180, timeout_s=6 * 3600)
    with open(
        os.path.join(HERE, "verify", "profiles", "fine.json"), encoding="utf-8"
    ) as handle:
        profile = json.load(handle)
    retried = 0
    for key, record in pending_records.items():
        task_id = record["active_id"]
        record["last_status"] = status.get(task_id)
        if record["last_status"] not in TERMINAL_TASK_STATUSES:
            continue
        try:
            fetched = sc.fetch_result(
                task_id,
                expected_revision=st["fine_solver_git_revision"],
                expected_library_revision=st["fine_pyaedt_library_git_revision"],
                expected_profile=profile["param_overrides"],
            )
        except sc.ResultFetchError as exc:
            record["outcome"] = "fetch_error"
            record["fetch_error"] = str(exc)
            save_state(st)
            raise RuntimeError(f"fine stdout unavailable for task {task_id}") from exc
        record["last_result_state"] = fetched.state
        candidate = st["final_candidates"][int(key)]
        result_matches = (
            fetched.state == sc.RESULT_VALID
            and sc.result_matches_params(
                fetched.result, candidate.get("fine_params"), required_keys=KEYS
            )
        )
        if result_matches:
            record["outcome"] = "valid"
            record["result"] = fetched.result
            save_state(st)
            continue
        if fetched.state == sc.RESULT_VALID:
            record["last_result_state"] = "candidate_result_identity_mismatch"
        if int(record.get("attempt", 0)) >= 1:
            # This is not a physical spec failure.  The smaller candidate
            # remains unverified and therefore blocks any minimum-volume PASS.
            record["outcome"] = "unverified"
            record["unverified_reason"] = record["last_result_state"]
        else:
            if not EXECUTE_SUBMISSIONS:
                raise RuntimeError(
                    "fine retry requires --execute; no larger candidate can be finalized"
                )
            rank = int(key)
            candidate = st["final_candidates"][rank]
            batch = int(st.get("fine_batch", 0))
            retry_name = (
                f"mft-final-fine-r{st['round']:02d}-b{batch:02d}-c{rank:02d}-retry"
            )
            retry_workdir = (
                f"mft_final_fine_r{st['round']:02d}_b{batch:02d}_c{rank:02d}_retry"
            )
            _require_runtime_deployment()
            new_task_id = sc.submit_verification(
                retry_name,
                retry_workdir,
                candidate["params"],
                profile,
                mem_mb=max(int(profile["mem_mb"]), 131072),
                cpus=profile["cpus"],
                solver_revision=st["fine_solver_git_revision"],
                library_revision=st["fine_pyaedt_library_git_revision"],
            )
            if new_task_id is None:
                record["outcome"] = "retry_submission_unknown"
                save_state(st)
                raise RuntimeError("fine retry task identity is unknown")
            record.update({
                "retry_id": new_task_id,
                "active_id": new_task_id,
                "attempt": 1,
                "outcome": "pending",
                "last_status": None,
            })
            retried += 1
        save_state(st)
    if retried:
        st["stage"] = "FINE_WAIT"
        return
    unverified = [
        key for key, record in records.items()
        if record.get("outcome") == "unverified"
    ]
    from optimization.geometry_metrics import bounding_box_lit
    from verify.finalize import candidate_identity_reasons, physical_spec_reasons
    passing_volumes = []
    for key, record in records.items():
        if record.get("outcome") != "valid":
            continue
        candidate = st["final_candidates"][int(key)]
        result = record.get("result") or {}
        if not (physical_spec_reasons(result)
                + candidate_identity_reasons(
                    result, candidate, expected_params_key="fine_params"
                )):
            try:
                volume = float(bounding_box_lit(result)[0])
                if not math.isfinite(volume) or volume <= 0:
                    raise ValueError("nonpositive fine volume")
                passing_volumes.append(volume)
                record["fine_volume_L"] = volume
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
    smallest_pass_volume = min(passing_volumes) if passing_volumes else None
    blocking_unverified = []
    for key in unverified:
        try:
            volume = float(st["final_candidates"][int(key)]["volume_L"])
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            blocking_unverified.append(key)
            continue
        if smallest_pass_volume is None or volume < smallest_pass_volume:
            blocking_unverified.append(key)
    if blocking_unverified:
        st["stage"] = "FINE_BLOCKED"
        st["fine_block_reason"] = (
            "smaller candidate fine FEA remained invalid/missing after one retry: "
            + ",".join(blocking_unverified)
        )
        print(f"[al] FINE BLOCKED: {st['fine_block_reason']}")
        return
    if any(
        record.get("outcome") not in ("valid", "unverified")
        for record in records.values()
    ):
        st["stage"] = "FINE_WAIT"
        return
    st["stage"] = "FINAL_REPORT"


def stage_final_report(st):
    _assert_training_invariants(st)
    from verify.finalize import (
        candidate_identity_reasons, physical_spec_reasons,
        write_final_artifacts,
    )

    records = st.get("fine_task_records") or {}
    fine_results = [
        records.get(str(rank), {}).get("result") or {}
        for rank in range(len(st.get("final_candidates") or []))
    ]
    for rank, candidate in enumerate(st.get("final_candidates") or []):
        record = records.get(str(rank), {})
        candidate["fine_task_id"] = record.get("active_id")
        candidate["fine_task_status"] = record.get("last_status")
        candidate["fine_attempt"] = record.get("attempt", 0)
    model_quality = st.get("model_quality_snapshot") or {}
    if (not model_quality.get("passed")
            or model_quality.get("training_run_id") != st.get("training_run_id")):
        raise RuntimeError("final report is blocked by a missing/failed model quality gate")
    current_pass = any(
        not (physical_spec_reasons(result)
             + candidate_identity_reasons(
                 result, candidate, expected_params_key="fine_params"
             ))
        for candidate, result in zip(st["final_candidates"], fine_results)
    )
    cursor = int(st.get("fine_queue_cursor", 0)) + len(st["final_candidates"])
    queue = st.get("fine_candidate_queue") or st["final_candidates"]
    if not current_pass and cursor < len(queue):
        # Every lower-volume candidate in prior batches has a terminal fine
        # failure.  Continue with the next volume-ordered batch before asking
        # NSGA-II for more designs.
        history = st.setdefault("fine_attempt_history", [])
        for rank, (candidate, result) in enumerate(zip(
                st["final_candidates"], fine_results)):
            record = records.get(str(rank), {})
            history.append({
                "candidate_digest": candidate["candidate_digest"],
                "volume_L": candidate["volume_L"],
                "passed": False,
                "reasons": (
                    physical_spec_reasons(result)
                    + candidate_identity_reasons(
                        result, candidate, expected_params_key="fine_params"
                    )
                ),
                "fine_result": result,
                "fine_task_id": record.get("active_id"),
                "fine_task_status": record.get("last_status"),
                "fine_attempt": record.get("attempt", 0),
                "solver_revision": st.get("fine_solver_git_revision"),
                "library_revision": st.get(
                    "fine_pyaedt_library_git_revision"
                ),
            })
        st["fine_queue_cursor"] = cursor
        st["fine_batch"] = int(st.get("fine_batch", 0)) + 1
        st["final_candidates"] = queue[cursor: cursor + 3]
        st["fine_task_records"] = {}
        st["stage"] = "FINE_SUBMIT"
        print("[al] fine batch failed; advancing to next minimum-volume candidates")
        return
    artifact = write_final_artifacts(
        OUTPUT_ROOT,
        st["final_candidates"],
        fine_results,
        model_quality,
        st["fine_solver_git_revision"],
        st["fine_pyaedt_library_git_revision"],
        prior_attempts=st.get("fine_attempt_history"),
    )
    st["final_verification"] = {
        "passed": artifact["passed"],
        "generated_at": artifact["generated_at"],
    }
    st["fine_attempt_history"] = artifact.get("fine_attempts", [])
    if artifact["passed"]:
        st["stage"] = "DONE"
        print("[al] FINAL FINE FEA VERIFIED")
    elif st["round"] >= SPEC["max_rounds"]:
        st["stage"] = "HARD_CAP"
    else:
        st["q_mult"] = min(st.get("q_mult", 1.0) * 1.25, 3.0)
        st["round"] += 1
        st["stage"] = "TRAIN"


STAGES = {"TRAIN": stage_train, "OPTIMIZE": stage_optimize, "SELECT": stage_select,
          "SUBMIT": stage_submit, "WAIT": stage_wait, "INGEST": stage_ingest,
          "CHECK": stage_check, "FINAL_SELECT": stage_final_select,
          "FINE_SUBMIT": stage_fine_submit, "FINE_WAIT": stage_fine_wait,
          "FINAL_REPORT": stage_final_report}


def main():
    global EXECUTE_SUBMISSIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--max-stages", type=int, default=200)
    ap.add_argument("--runtime-root", default=HERE)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--solver-revision", default=None)
    ap.add_argument("--library-revision", default=None)
    ap.add_argument("--library-root", default=None)
    ap.add_argument(
        "--execute", action="store_true",
        help="allow standard/fine FEA task submission; submission is disabled by default",
    )
    args = ap.parse_args()

    for label, revision in (
        ("solver", args.solver_revision), ("library", args.library_revision),
    ):
        value = str(revision or "").strip().lower()
        if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
            ap.error(f"{label} revision must be an explicit full 40-character SHA")
        if label == "solver":
            args.solver_revision = value
        else:
            args.library_revision = value

    configure_runtime(
        runtime_root=args.runtime_root,
        dataset=args.dataset,
        output_root=args.output_root,
        registry=args.registry,
        solver_revision=args.solver_revision,
        library_revision=args.library_revision,
        library_root=args.library_root,
    )
    EXECUTE_SUBMISSIONS = bool(args.execute)
    if EXECUTE_SUBMISSIONS and not (
            PINNED_LIBRARY_ROOT
            or os.environ.get("MFT_PYAEDT_LIBRARY_ROOT", "").strip()):
        ap.error("--execute requires --library-root or MFT_PYAEDT_LIBRARY_ROOT")
    if args.reset and os.path.isfile(STATE_PATH):
        os.remove(STATE_PATH)
    st = load_state()
    _bind_runtime_identity(st)
    for _ in range(args.max_stages):
        if st["stage"] in ("DONE", "HARD_CAP", "FINE_BLOCKED"):
            print(f"[al] {st['stage'].lower()}. history:")
            for h in st["history"]:
                print("  ", h)
            break
        print(f"\n[al] === round {st['round']} / stage {st['stage']} ===")
        STAGES[st["stage"]](st)
        save_state(st)


if __name__ == "__main__":
    main()
