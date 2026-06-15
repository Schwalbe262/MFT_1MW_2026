from __future__ import annotations

import hashlib
import random
from typing import Any


def evaluate(candidates: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        params = candidate.parameters
        seed = int(hashlib.sha256(candidate.candidate_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        n1 = float(params["N1"])
        w1 = float(params["w1"])
        l1 = float(params["l1"])
        l2 = float(params["l2"])
        h1 = float(params["h1"])
        window_ratio = float(params["window_ratio"])
        fill = (float(params["wff1"]) + float(params["wff2"])) / 2
        k = max(0.05, min(0.98, 0.42 + 0.32 * fill - 0.001 * abs(window_ratio - 0.5) + rng.uniform(-0.02, 0.02)))
        lmt = 0.018 * n1 * n1 * (w1 + h1) / max(1.0, l1 + l2) * k
        leakage = 0.015 * (l1 + l2) / max(1.0, w1) + rng.uniform(0.0, 0.2)
        tx_loss = 450 + 0.8 * w1 + 2.0 * h1 / max(1.0, n1) + rng.uniform(-30, 30)
        rx_loss = 480 + 0.35 * (l1 + l2) + rng.uniform(-30, 30)
        row = {
            "candidate_id": candidate.candidate_id,
            "loop": candidate.loop,
            "status": "completed",
            "Lmt": f"{lmt:.8g}",
            "Llt": f"{leakage:.8g}",
            "Llr": f"{leakage * 1.1:.8g}",
            "k": f"{k:.8g}",
            "Tx_loss": f"{tx_loss:.8g}",
            "Rx_loss": f"{rx_loss:.8g}",
        }
        row.update(params)
        rows.append(row)
    return rows
