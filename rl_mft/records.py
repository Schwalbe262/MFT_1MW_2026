from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS_DIR = Path("rl_runs")
STATE_PATH = RUNS_DIR / "state.json"
TOKEN_USAGE_PATH = RUNS_DIR / "token_usage.jsonl"
NOTE_PATH = Path("note.md")
INSIGHT_PATH = Path("insight.md")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LoopSummary:
    loop: int
    backend: str
    batch_size: int
    status: str
    started_at: str
    finished_at: str | None = None
    job_ids: list[str] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    best_reward: float | None = None
    best_candidate_id: str | None = None
    message: str = ""


def ensure_run_dirs() -> None:
    RUNS_DIR.mkdir(exist_ok=True)


def loop_dir(loop: int) -> Path:
    path = RUNS_DIR / f"loop_{loop:04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_state() -> dict[str, Any]:
    ensure_run_dirs()
    if not STATE_PATH.exists():
        return {
            "current_loop": 0,
            "best_reward": None,
            "best_candidate_id": None,
            "best_parameters": None,
            "live_best_reward": None,
            "live_best_candidate_id": None,
            "live_best_parameters": None,
            "live_elites": [],
            "loops": [],
            "elites": [],
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    ensure_run_dirs()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def write_candidates(path: Path, candidates: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_note(summary: LoopSummary) -> None:
    text = [
        "",
        f"## Loop {summary.loop} - {summary.started_at}",
        "",
        f"- backend: {summary.backend}",
        f"- status: {summary.status}",
        f"- batch_size: {summary.batch_size}",
        f"- completed: {summary.completed}",
        f"- failed: {summary.failed}",
        f"- job_ids: {', '.join(summary.job_ids) if summary.job_ids else '-'}",
        f"- best_reward: {summary.best_reward if summary.best_reward is not None else '-'}",
        f"- best_candidate_id: {summary.best_candidate_id or '-'}",
    ]
    if summary.finished_at:
        text.append(f"- finished_at: {summary.finished_at}")
    if summary.message:
        text.append(f"- message: {summary.message}")
    NOTE_PATH.write_text(NOTE_PATH.read_text(encoding="utf-8") + "\n".join(text) + "\n", encoding="utf-8")


def append_insight(loop: int, candidate_id: str, reward: float, parameters: dict[str, Any], label: str = "improved_candidate") -> None:
    compact = ", ".join(f"{key}={parameters[key]}" for key in sorted(parameters) if key in {"N1", "N2", "w1", "l1", "l2", "h1", "window_ratio", "wff1", "wff2"})
    text = (
        f"\n## {utc_now()} - Loop {loop}\n\n"
        f"- {label}: {candidate_id}\n"
        f"- reward: {reward:.6g}\n"
        f"- key_parameters: {compact}\n"
    )
    INSIGHT_PATH.write_text(INSIGHT_PATH.read_text(encoding="utf-8") + text, encoding="utf-8")


def append_token_usage(
    provider: str,
    project: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
    note: str = "",
    reset_cycle: str = "",
) -> None:
    ensure_run_dirs()
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens
    row = {
        "recorded_at": utc_now(),
        "provider": provider,
        "project": project,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total),
        "reset_cycle": reset_cycle,
        "note": note,
    }
    with TOKEN_USAGE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_token_usage() -> list[dict[str, Any]]:
    ensure_run_dirs()
    if not TOKEN_USAGE_PATH.exists():
        return []
    return [json.loads(line) for line in TOKEN_USAGE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
