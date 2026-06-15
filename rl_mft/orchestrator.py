from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from rl_mft.mock_backend import evaluate as mock_evaluate
from rl_mft.parameters import coerce_partial_parameter_types, propose_batch
from rl_mft.records import (
    LoopSummary,
    OUTPUT_COLUMNS,
    append_insight,
    append_note,
    load_state,
    loop_dir,
    read_results,
    save_state,
    utc_now,
    write_candidates,
    write_results,
)
from rl_mft.reward import attach_rewards, load_reward_config
from rl_mft.scheduler_client import SlurmSchedulerConfig, fetch_remote_result_csv, submit_dynamic_batch, wait_for_jobs


def best_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("reward", "-inf")))


def parameter_subset(row: dict, template: dict) -> dict:
    return coerce_partial_parameter_types({key: row[key] for key in template if key in row})


def output_subset(row: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in OUTPUT_COLUMNS:
        if key in row and row[key] not in {"", None}:
            try:
                out[key] = float(row[key])
            except (TypeError, ValueError):
                out[key] = row[key]
    return out


def rows_for_candidates(rows: list[dict], candidates: list[Any]) -> list[dict]:
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    return [row for row in rows if row.get("candidate_id") in candidate_ids]


def run_loop(args: argparse.Namespace) -> LoopSummary:
    state = load_state()
    loop = int(state.get("current_loop", 0)) + 1
    summary = LoopSummary(loop=loop, backend=args.backend, batch_size=args.batch_size, status="running", started_at=utc_now())
    ldir = loop_dir(loop)
    if args.backend == "slurm":
        live_best = state.get("live_best_parameters")
        elite_source = [live_best] if live_best else state.get("live_elites") or state.get("elites") or []
    else:
        elite_source = state.get("elites") or []
    candidates = propose_batch(loop, args.batch_size, elites=elite_source)
    candidates_path = ldir / "candidates.jsonl"
    results_path = ldir / "results.csv"
    reward_config = load_reward_config(args.reward_config)
    write_candidates(candidates_path, candidates)

    try:
        if args.backend == "local-mock":
            rows = attach_rewards(mock_evaluate(candidates), reward_config)
            write_results(results_path, rows)
            summary.completed = len(rows)
        else:
            config = SlurmSchedulerConfig(
                base_url=args.scheduler_url,
                remote_path=args.remote_path,
                entrypoint=args.entrypoint,
                partition=args.partition,
                time_limit=args.time_limit,
                cpus_per_simulation=args.cpus_per_simulation,
                mem_per_simulation_gb=args.mem_per_simulation_gb,
                max_workers_per_job=args.max_workers_per_job,
                max_new_jobs=args.max_new_jobs,
                poll_seconds=args.poll_seconds,
            )
            summary.job_ids = submit_dynamic_batch(
                config,
                loop,
                len(candidates),
                remote_candidates_jsonl=args.remote_candidates_jsonl,
                candidates_jsonl_content="" if args.remote_candidates_jsonl else candidates_path.read_text(encoding="utf-8"),
            )
            jobs = wait_for_jobs(config, summary.job_ids) if args.wait else []
            summary.completed = sum(1 for job in jobs if job.get("status") == "completed")
            summary.failed = sum(1 for job in jobs if job.get("status") in {"failed", "cancelled"})
            fetched = 0
            if args.wait and summary.job_ids:
                fetched = 1 if fetch_remote_result_csv(config, summary.job_ids[0], results_path, remote_file=args.remote_result_file) else 0
            summary.message = f"Slurm jobs finished; fetched shared result file: {fetched}/1."
            rows = read_results(results_path)
            filtered_rows = rows_for_candidates(rows, candidates)
            if len(filtered_rows) != len(rows):
                summary.message += f" Discarded {len(rows) - len(filtered_rows)} stale result row(s)."
            rows = filtered_rows
            rows = attach_rewards(rows, reward_config) if rows else []
            if rows:
                write_results(results_path, rows)
                summary.completed = len(rows)
                summary.failed = max(0, args.batch_size - len(rows))
            elif jobs and all(job.get("status") == "completed" for job in jobs):
                summary.failed = args.batch_size
                summary.completed = 0
                summary.message += " No result rows were collected; treating loop as failed."

        evaluated_rows = rows_for_candidates(read_results(results_path), candidates)
        row = best_row(evaluated_rows)
        previous_best = state.get("best_reward")
        if row:
            reward = float(row["reward"])
            summary.best_reward = reward
            summary.best_candidate_id = row.get("candidate_id")
            parameters = parameter_subset(row, candidates[0].parameters)
            outputs = output_subset(row)
            summary.best_outputs = outputs
            if args.backend == "slurm":
                previous_live_best = state.get("live_best_reward")
                live_improved = previous_live_best is not None and reward > float(previous_live_best)
                live_baseline = previous_live_best is None
                if live_baseline:
                    append_insight(loop, summary.best_candidate_id or "", reward, parameters, outputs, label="live_baseline_candidate")
                elif live_improved:
                    append_insight(loop, summary.best_candidate_id or "", reward, parameters, outputs, label="live_improved_candidate")
                if live_baseline or live_improved:
                    state["live_best_reward"] = reward
                    state["live_best_candidate_id"] = summary.best_candidate_id
                    state["live_best_parameters"] = parameters
                    state["live_best_outputs"] = outputs
                summary.message = (summary.message + " " if summary.message else "") + (
                    "Live baseline established." if live_baseline else f"Live improved: {live_improved}."
                )
            improved = previous_best is None or reward > float(previous_best)
            if improved and summary.best_candidate_id:
                if args.backend != "slurm":
                    append_insight(loop, summary.best_candidate_id, reward, parameters, outputs)
                state["best_reward"] = reward
                state["best_candidate_id"] = summary.best_candidate_id
                state["best_parameters"] = parameters
                state["best_outputs"] = outputs
            ranked = sorted(evaluated_rows, key=lambda item: float(item.get("reward", "-inf")), reverse=True)
            ranked_parameters = [parameter_subset(row, candidates[0].parameters) for row in ranked[: max(1, args.elite_count)]]
            if args.backend == "slurm":
                if live_baseline or live_improved:
                    state["live_elites"] = ranked_parameters
            else:
                state["elites"] = ranked_parameters
            state["recent_evaluations"] = ranked[: min(50, len(ranked))]
        state["current_loop"] = loop
        state["failure_rate"] = (summary.failed / max(1, args.batch_size))
        state.setdefault("loops", []).append(asdict(summary))
        summary.status = "completed"
    except Exception as exc:
        summary.status = "failed"
        summary.message = str(exc)
        summary.failed = args.batch_size - summary.completed
        state["current_loop"] = loop
        state.setdefault("loops", []).append(asdict(summary))
        raise
    finally:
        summary.finished_at = utc_now()
        state["loops"][-1] = asdict(summary)
        save_state(state)
        append_note(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one MFT RL optimization loop.")
    parser.add_argument("--backend", choices=["local-mock", "slurm"], default="local-mock")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=5)
    parser.add_argument("--reward-config", default="rl_reward_config.json")
    parser.add_argument("--scheduler-url", default="http://127.0.0.1:8000")
    parser.add_argument("--remote-path", default="~/MFT_1MW_2026")
    parser.add_argument("--remote-candidates-jsonl", default="", help="Optional existing candidate JSONL path on the Slurm remote. If omitted, candidates are embedded into the sbatch env setup.")
    parser.add_argument("--remote-result-file", default="simulation_results.csv")
    parser.add_argument("--entrypoint", default="rl_mft/fea_worker.py")
    parser.add_argument("--partition", default="auto")
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--cpus-per-simulation", type=int, default=4)
    parser.add_argument("--mem-per-simulation-gb", type=float, default=8.0)
    parser.add_argument("--max-workers-per-job", type=int, default=16)
    parser.add_argument("--max-new-jobs", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_loop(args)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
