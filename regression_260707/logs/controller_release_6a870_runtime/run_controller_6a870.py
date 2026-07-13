"""Y-only production launcher for reviewed controller release 6a870392.

This is a runtime artifact, not a source replacement.  It authenticates the
nine restored files against the reviewed release, binds every mutable path and
lock to Y:, and then delegates to the no-autopause controller.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

from filelock import FileLock


PROJECT_ROOT = Path(r"Y:\git\MFT_1MW_2026")
REGRESSION_ROOT = PROJECT_ROOT / "regression_260707"
RUNTIME_ROOT = REGRESSION_ROOT / "logs" / "controller_release_6a870_runtime"
LOCK_ROOT = RUNTIME_ROOT / "locks"
REVIEWED_RELEASE = "6a870392a910d8f964c7dae5ca2be89e23021171"
PLAN_SHA = "b24e2a9b00caa22bbec8793f4dbd99de51362fac87f9e9509358610abe9982d0"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED = {
    "campaign/_continuous_refill_b171c7c.py": "76AC1750A1CBBB7C6807BBA18CA4B5BED529DE14DB1567124E075A7AB08BCBE5",
    "campaign/_adopted_refill_sha688c6f9.py": "E7C2D2ACBFEFEEDA96CAEF28CCF59E5B02278ACA607E856FB1580A7CBEFA8EA4",
    "campaign/_rolling_recycle_prebinding_260712.py": "4BB5282C3DB4B59211934D13851120021651B9427D8307E68B9C870795E22C1B",
    "campaign/_submit_production300_b171c7c.py": "12ADADC7FFDD785512F284BA44111B10C83E6AD77636C5325430E2F3AFF6D704",
    "campaign/tests/test_adopted_refill.py": "002F5AFBFCB7B00BF19320B98F05B28E1EE6ED7083C6FB64BDB890E7118797FE",
    "campaign/tests/test_continuous_refill_incidents.py": "EAF0B26C682AA6362F2FD560B766B03E091E8FA70300723EFAA345E21BFBA9C6",
    "campaign/feeder.py": "6DCC8976B905F2961C6F9E89945BD5B38DD903A6995F0F71D147D948A2F36093",
    "campaign/rapid_campaign.py": "ABF266A0845D9311F8D29F6832DD39DBDA7F7579B104FB5285B85C5161C4B18F",
    "verify/scheduler_client.py": "9BA34BA3F51841C98FB0E7AB18754BF195A391AE087494AAAC977477C8FF900B",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _authenticate() -> None:
    if any("pool400" in arg.casefold() for arg in sys.argv[1:]):
        raise RuntimeError("pool400 authorization is forbidden")
    for relative, expected in EXPECTED.items():
        path = REGRESSION_ROOT / relative
        actual = _sha(path)
        if actual != expected:
            raise RuntimeError(
                f"reviewed source hash mismatch: {relative} {actual} != {expected}"
            )


def main() -> int:
    _authenticate()
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    os.environ["GIT_CONFIG_VALUE_0"] = "*"

    from regression_260707.verify import scheduler_client

    scheduler_client.CAMPAIGN_MUTATION_LOCK_PATH = (
        LOCK_ROOT / "campaign-mutation.lock"
    )
    from regression_260707.campaign import _continuous_refill_b171c7c as controller

    controller.STATE_PATH = (
        REGRESSION_ROOT / "campaign" / "continuous_refill_b171c7c_state.json"
    )
    controller.FEEDER_STATE_PATH = (
        REGRESSION_ROOT
        / "campaign"
        / "continuous_refill_b171c7c_feeder_state.json"
    )
    controller.CYCLE_ROOT = (
        REGRESSION_ROOT
        / "campaign"
        / "pilot_manifests"
        / "continuous-refill-sb171c7c-le6b9b9d"
    )
    controller.TARGET_TRANSITION_ROOT = controller.CYCLE_ROOT / "target-transitions"
    controller.STRICT_STATUS_PATH = (
        REGRESSION_ROOT / "training" / "strict_data_status.json"
    )
    controller.feeder.STATE = str(controller.FEEDER_STATE_PATH)
    controller.feeder.TRAIN_PARQUET = str(
        REGRESSION_ROOT / "data" / "dataset" / "train.parquet"
    )
    controller.feeder.COLLECT_CACHE = str(
        REGRESSION_ROOT / "data" / "dataset" / "collect_cache.json"
    )

    if str(scheduler_client.CAMPAIGN_MUTATION_LOCK_PATH).upper().startswith("C:\\"):
        raise RuntimeError("C: mutation lock is forbidden")
    if controller.TARGET_ACTIVE != 300:
        raise RuntimeError(f"target drifted from 300: {controller.TARGET_ACTIVE}")
    if scheduler_client.MFT_PROJECT != "MFT_1MW_2026v1":
        raise RuntimeError(f"project drift: {scheduler_client.MFT_PROJECT}")

    loop_lock = FileLock(str(LOCK_ROOT / "controller-loop.lock"), timeout=0)
    with loop_lock:
        return int(controller.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
