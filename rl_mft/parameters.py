from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


PARAMETER_COLUMNS = [
    "N1",
    "N2",
    "N1_main",
    "N1_side",
    "N2_main",
    "N2_side",
    "w1",
    "l1",
    "l2",
    "h1",
    "cc_w2c_space_x",
    "w2c_w1c_space_x",
    "w1c_w2s_space_x",
    "w2s_w1s_space_x",
    "w1s_cs_space_x",
    "cc_w2c_space_y",
    "w2c_w1c_space_y",
    "cs_w1s_space_y",
    "w1s_w2s_space_y",
    "window_ratio",
    "wh1",
    "wh2",
    "wff1",
    "wff2",
]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    loop: int
    index: int
    parameters: dict[str, Any]


def dataframe_row_to_dict(df: Any) -> dict[str, Any]:
    row = df.iloc[0].to_dict()
    out: dict[str, Any] = {}
    for key in PARAMETER_COLUMNS:
        value = row[key]
        if hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


def _load_existing_parameter_tools() -> tuple[Any, Any] | tuple[None, None]:
    try:
        from module.input_parameter import create_input_parameter, validation_check
    except Exception:
        return None, None
    return create_input_parameter, validation_check


def _fallback_parameters() -> dict[str, Any]:
    n1 = random.randint(5, 10)
    n2 = n1 * 10
    n1_side = round(n1 * random.uniform(0.0, 0.5))
    n1_main = n1 - n1_side
    n2_side = round(n2 * random.uniform(0.0, 0.8))
    n2_main = n2 - n2_side
    l1 = random.randint(40, 100)
    total_length = random.randint(500, 1200)
    total_height = random.randint(500, 1000)
    return {
        "N1": n1,
        "N2": n2,
        "N1_main": n1_main,
        "N1_side": n1_side,
        "N2_main": n2_main,
        "N2_side": n2_side,
        "w1": random.randint(200, 800),
        "l1": l1,
        "l2": (total_length - 4 * l1) / 2,
        "h1": total_height - 2 * l1,
        "cc_w2c_space_x": round(random.uniform(10, 50), 1),
        "w2c_w1c_space_x": round(random.uniform(10, 50), 1),
        "w1c_w2s_space_x": round(random.uniform(10, 100), 1),
        "w2s_w1s_space_x": round(random.uniform(10, 50), 1),
        "w1s_cs_space_x": round(random.uniform(10, 50), 1),
        "cc_w2c_space_y": round(random.uniform(10, 50), 1),
        "w2c_w1c_space_y": round(random.uniform(10, 50), 1),
        "cs_w1s_space_y": round(random.uniform(10, 50), 1),
        "w1s_w2s_space_y": round(random.uniform(10, 50), 1),
        "window_ratio": round(random.uniform(0.3, 0.7), 2),
        "wh1": round(random.uniform(0.8, 0.95), 2),
        "wh2": round(random.uniform(0.5, 0.95), 2),
        "wff1": round(random.uniform(0.4, 0.8), 2),
        "wff2": round(random.uniform(0.4, 0.75), 2),
    }


def is_valid(parameters: dict[str, Any]) -> bool:
    create_input_parameter, validation_check = _load_existing_parameter_tools()
    if create_input_parameter is None or validation_check is None:
        return float(parameters["l2"]) > 0 and float(parameters["h1"]) > 0
    df = create_input_parameter([parameters[column] for column in PARAMETER_COLUMNS])
    ok, _ = validation_check(df)
    return bool(ok)


def random_candidate(loop: int, index: int) -> Candidate:
    create_input_parameter, validation_check = _load_existing_parameter_tools()
    if create_input_parameter is not None and validation_check is not None:
        while True:
            df = create_input_parameter()
            ok, _ = validation_check(df)
            if ok:
                parameters = dataframe_row_to_dict(df)
                break
    else:
        while True:
            parameters = _fallback_parameters()
            if is_valid(parameters):
                break
    return Candidate(
        candidate_id=f"L{loop:04d}-C{index:04d}",
        loop=loop,
        index=index,
        parameters=parameters,
    )


def mutate_candidate(base: dict[str, Any], loop: int, index: int, scale: float = 0.08) -> Candidate:
    values = dict(base)
    for key in ["w1", "l1", "l2", "h1"]:
        values[key] = max(1, round(float(values[key]) * random.uniform(1 - scale, 1 + scale), 3))
    for key in [
        "cc_w2c_space_x",
        "w2c_w1c_space_x",
        "w1c_w2s_space_x",
        "w2s_w1s_space_x",
        "w1s_cs_space_x",
        "cc_w2c_space_y",
        "w2c_w1c_space_y",
        "cs_w1s_space_y",
        "w1s_w2s_space_y",
    ]:
        values[key] = max(0.1, round(float(values[key]) * random.uniform(1 - scale, 1 + scale), 3))
    for key in ["window_ratio", "wh1", "wh2", "wff1", "wff2"]:
        values[key] = min(0.98, max(0.05, round(float(values[key]) * random.uniform(1 - scale, 1 + scale), 3)))
    if not is_valid(values):
        return random_candidate(loop, index)
    return Candidate(candidate_id=f"L{loop:04d}-C{index:04d}", loop=loop, index=index, parameters=values)


def propose_batch(loop: int, batch_size: int, elites: list[dict[str, Any]] | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    elites = elites or []
    for index in range(1, batch_size + 1):
        if elites and index <= max(1, batch_size // 2):
            base = random.choice(elites)
            candidates.append(mutate_candidate(base, loop, index))
        else:
            candidates.append(random_candidate(loop, index))
    return candidates
