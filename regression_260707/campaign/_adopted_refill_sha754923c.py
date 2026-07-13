"""Adopt the sealed SHA754923c replacement fleet for guarded refill."""
from __future__ import annotations

import _adopted_refill_sha688c6f9 as controller


SOLVER = "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
MANIFEST_SHA256 = "f1490f2cda497c9475fe079fb0a04e5adb7686c6f4c99ae28a0f946a918319a8"


controller.SOLVER = SOLVER
controller.LIBRARY = LIBRARY
controller.SEED = 260710
controller.PREFIX = f"mft-camp-s{SOLVER[:7]}-l{LIBRARY[:7]}-"
controller.INITIAL_COUNT = 250
controller.INITIAL_FIRST_ID = 27755
controller.INITIAL_LAST_ID = 28004
controller.INITIAL_FIRST_SERIAL = 17362
controller.INITIAL_LAST_SERIAL = 17611
controller.INITIAL_CURSOR_START = 1843
controller.INITIAL_CURSOR_END = 2795
controller.INITIAL_LAST_RAW_INDEX = 2794
controller.MANIFEST_SHA256 = MANIFEST_SHA256
controller.MANIFEST_PATH = controller.HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.json"
)
controller.SUBMISSION_JOURNAL_PATH = controller.MANIFEST_PATH.with_name(
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.journal.json"
)
controller.STATE_PATH = controller.HERE / "adopted_refill_754923c_state.json"
controller.FEEDER_STATE_PATH = controller.HERE / (
    "adopted_refill_754923c_feeder_state.json"
)
controller.CYCLE_ROOT = controller.HERE / "pilot_manifests" / (
    "adopted-refill-s754923c-le6b9b9d"
)


if __name__ == "__main__":
    raise SystemExit(controller.main())
