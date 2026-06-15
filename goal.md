# MFT 1MW RL Design Research Goal

## Objective

Use reinforcement learning to search for improved 1 MW MFT geometry designs, with actual design claims based only on Slurm/FEA results.

The current codebase already has a working RL scaffold. The next goal is to run real FEA-backed loops and confirm that the loop can produce a design that improves over the live FEA baseline.

## Research Definition

- State: current loop number, recent evaluated designs, failure rate, live best design, live best reward, and optional surrogate/model history.
- Action: propose one or more valid MFT geometry candidates from the existing validated design-variable space.
- Environment: Ansys/PyAEDT FEA executed through Slurm using the `slurm_scheduler` packed allocation model.
- Reward: configurable weighted score from coupling/inductance benefits and loss/leakage penalties, defined in `rl_reward_config.json`.
- Episode: one optimization loop. For live validation, the default episode size is one candidate to isolate failures and control resource use.

## Live Validation Strategy

1. Run one Slurm/FEA candidate as the live baseline.
2. Continue running one-candidate Slurm/FEA loops.
3. A live loop succeeds when its reward is greater than the previous live best reward.
4. Continue until a live improvement is observed.
5. Mock results are allowed only for software checks and must not be used as evidence of real design improvement.

## Slurm Execution Model

Use the existing `slurm_scheduler` service at `http://127.0.0.1:8000`.

- Submit through `dynamic_packed_srun`.
- Use `SIMULATION_ID` to select the candidate row from the loop candidate JSONL.
- Run FEA through `rl_mft/fea_worker.py`, which calls `run_simulation_260514.run_one_loop(param=...)`.
- Fetch the completed shared `simulation_results.csv` through the scheduler remote-file API.
- Score fetched rows locally and update live state.

This remains intentionally based on packed Slurm allocations rather than `sattach`.

## Required Records

- `note.md`: append one entry for every loop attempt.
- `insight.md`: append only when a live best design improves or when explicitly recording the first live baseline.
- `rl_runs/state.json`: machine-readable state for continuation and dashboard display.
- `rl_runs/token_usage.jsonl`: token usage records for dashboard chart/table tracking.
- `rl_runs/loop_*/candidates.jsonl`: candidate inputs for each loop.
- `rl_runs/loop_*/results.csv`: mock or live evaluated results for each loop.

## Success Criteria

- Preflight tests pass for the RL code and Slurm scheduler integration.
- The dashboard at `http://127.0.0.1:8010` shows loop state, live best state, failure rate, and token usage graph/table.
- At least one Slurm/FEA loop completes and writes a real `simulation_results.csv`.
- The first live result establishes `live_best_reward`.
- A later live loop improves on `live_best_reward`, updates `rl_runs/state.json`, and appends an improvement to `insight.md`.

## Repository And Runtime Policy

- Work on the local WSL clone in branch `rl-design`.
- Treat `Y:\git\MFT_1MW_2026` as an SFTP/NAS reference or sync target, not as the active Git working tree.
- Keep cluster credentials, local scheduler config, Slurm job IDs from unrelated manual runs, and generated FEA artifacts out of Git.
