"""Exact specialization of the reviewed recovery4 submitter for SHA b171c7c."""
from __future__ import annotations

import _submit_thermal_recovery4 as engine


engine.SOLVER = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
engine.PLAN_SHA256 = "3e453deb61137c2d29c13bbbe8d5117b4c4111e5ea7e255d37dfd0d5e4444af5"
engine.PLAN_PATH = engine.HERE / "pilot_manifests" / (
    "thermal-recovery4-sb171c7c-le6b9b9d.json"
)
engine.PARTIAL_PATH = engine.PLAN_PATH.with_name(
    engine.PLAN_PATH.stem + ".submission.partial.json"
)
engine.FINAL_PATH = engine.PLAN_PATH.with_name(
    engine.PLAN_PATH.stem + ".submission.json"
)
engine.PREFIX = "mft-recovery4-sb171c7c-le6b9b9d-"


if __name__ == "__main__":
    raise SystemExit(engine.main())
