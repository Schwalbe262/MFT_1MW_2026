from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlurmSchedulerConfig:
    base_url: str = "http://127.0.0.1:8000"
    remote_path: str = "~/MFT_1MW_2026"
    entrypoint: str = "rl_mft/fea_worker.py"
    env_setup: str = "source ~/miniconda3/etc/profile.d/conda.sh\nconda activate pyaedt2026v1\nmodule load ansys-electronics/v252"
    partition: str = "auto"
    time_limit: str = "12:00:00"
    cpus_per_simulation: int = 4
    mem_per_simulation_gb: float = 8.0
    max_workers_per_job: int = 16
    max_new_jobs: int = 4
    poll_seconds: int = 30


def _request_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def submit_dynamic_batch(
    config: SlurmSchedulerConfig,
    loop: int,
    candidate_count: int,
    remote_candidates_jsonl: str = "",
    candidates_jsonl_content: str = "",
) -> list[str]:
    candidate_file = remote_candidates_jsonl or f".rl_candidates_loop_{loop:04d}.jsonl"
    env_setup = config.env_setup
    if candidates_jsonl_content:
        env_setup += f"\ncat > {candidate_file} <<'MFT_RL_CANDIDATES_EOF'\n"
        env_setup += candidates_jsonl_content.rstrip() + "\n"
        env_setup += "MFT_RL_CANDIDATES_EOF"
    env_setup += f"\nexport MFT_RL_CANDIDATES_JSONL={candidate_file}"
    form = {
        "job_mode": "dynamic_packed_srun",
        "entrypoint": config.entrypoint,
        "arguments": "",
        "env_setup": env_setup,
        "partition": config.partition,
        "time_limit": config.time_limit,
        "job_name": f"mft-rl-loop-{loop:04d}",
        "remote_path": config.remote_path,
        "total_simulations": str(candidate_count),
        "simulations_per_job": str(candidate_count),
        "cpus_per_simulation": str(config.cpus_per_simulation),
        "mem_per_simulation_gb": str(config.mem_per_simulation_gb),
        "max_workers_per_job": str(config.max_workers_per_job),
        "max_new_jobs": str(config.max_new_jobs),
        "memory": f"{max(4, int(candidate_count * config.mem_per_simulation_gb))}G",
        "gpus": "0",
    }
    data = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(f"{config.base_url.rstrip('/')}/jobs", data=data, method="POST")
    urllib.request.urlopen(request, timeout=30).read()
    jobs = _request_json(f"{config.base_url.rstrip('/')}/api/jobs")
    prefix = f"mft-rl-loop-{loop:04d}"
    return [str(job["id"]) for job in jobs if str(job.get("job_name", "")).startswith(prefix)]


def wait_for_jobs(config: SlurmSchedulerConfig, job_ids: list[str]) -> list[dict[str, Any]]:
    terminal = {"completed", "failed", "cancelled"}
    while True:
        jobs = _request_json(f"{config.base_url.rstrip('/')}/api/jobs")
        selected = [job for job in jobs if str(job.get("id")) in set(job_ids)]
        if selected and all(job.get("status") in terminal for job in selected):
            return selected
        time.sleep(config.poll_seconds)


def fetch_remote_result_csv(config: SlurmSchedulerConfig, job_id: str, output_path: Path, remote_file: str = "simulation_results.csv") -> bool:
    query = urllib.parse.urlencode({"path": remote_file, "base": "remote_path"})
    url = f"{config.base_url.rstrip('/')}/api/jobs/{urllib.parse.quote(str(job_id))}/remote-file?{query}"
    try:
        text = _request_text(url)
    except Exception:
        return False
    if not text.strip():
        return False
    if output_path.exists() and output_path.stat().st_size > 0:
        existing = output_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) > 1:
            output_path.write_text(existing.rstrip() + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    else:
        output_path.write_text(text, encoding="utf-8")
    return True
