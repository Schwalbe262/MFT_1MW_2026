from __future__ import annotations

import json
import os
from pathlib import Path

from rl_mft.parameters import PARAMETER_COLUMNS


def load_candidate() -> dict:
    path = os.environ.get("MFT_RL_CANDIDATES_JSONL")
    if not path:
        raise RuntimeError("MFT_RL_CANDIDATES_JSONL is not set")
    sim_id = int(os.environ.get("SIMULATION_ID", "1"))
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    index = sim_id - 1
    if index < 0 or index >= len(rows):
        raise IndexError(f"SIMULATION_ID={sim_id} is outside candidate range 1..{len(rows)}")
    return rows[index]


def main() -> None:
    candidate = load_candidate()
    parameters = candidate["parameters"]
    os.environ["MFT_RL_CANDIDATE_ID"] = candidate["candidate_id"]
    os.environ["MFT_RL_LOOP"] = str(candidate["loop"])
    os.environ["MFT_RL_CANDIDATE_INDEX"] = str(candidate["index"])
    param_list = [parameters[column] for column in PARAMETER_COLUMNS]

    from run_simulation_260514 import run_one_loop

    run_one_loop(param=param_list)


if __name__ == "__main__":
    main()
