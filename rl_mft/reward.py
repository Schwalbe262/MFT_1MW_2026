from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    maximize: dict[str, float] = field(default_factory=lambda: {"Lmt": 2.0, "k": 100.0})
    minimize: dict[str, float] = field(default_factory=lambda: {"Tx_loss": 0.02, "Rx_loss": 0.02, "Llt": 0.5, "Llr": 0.5})
    targets: dict[str, dict[str, float]] = field(default_factory=dict)
    failed_reward: float = -1000000000.0


def load_reward_config(path: str | Path | None = None) -> RewardConfig:
    if not path:
        return RewardConfig()
    config_path = Path(path)
    if not config_path.exists():
        return RewardConfig()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return RewardConfig(
        maximize={str(key): float(value) for key, value in raw.get("maximize", {}).items()},
        minimize={str(key): float(value) for key, value in raw.get("minimize", {}).items()},
        targets={
            str(key): {"value": float(value["value"]), "weight": float(value["weight"])}
            for key, value in raw.get("targets", {}).items()
        },
        failed_reward=float(raw.get("failed_reward", -1000000000.0)),
    )


def to_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_reward(row: dict[str, Any], config: RewardConfig | None = None) -> float:
    config = config or RewardConfig()
    reward = 0.0
    for column, weight in config.maximize.items():
        reward += weight * to_float(row, column)
    for column, weight in config.minimize.items():
        reward -= weight * to_float(row, column)
    for column, spec in config.targets.items():
        target = float(spec["value"])
        weight = float(spec["weight"])
        reward -= weight * abs(to_float(row, column) - target)
    return reward


def attach_rewards(rows: list[dict[str, Any]], config: RewardConfig | None = None) -> list[dict[str, Any]]:
    config = config or RewardConfig()
    out = []
    for row in rows:
        enriched = dict(row)
        if str(enriched.get("status", "completed")) == "failed":
            enriched["reward"] = f"{config.failed_reward:.12g}"
        else:
            enriched["reward"] = f"{compute_reward(enriched, config):.12g}"
        out.append(enriched)
    return out
