"""Collect old-v1 and split-temperature-v2 fixed-5T/Lm2mH NSGA-II.

The aggregate includes the authenticated old 16-seed campaign and 512 fresh
split-temperature seeds.  Exact-seed retry manifests replace failed tasks
before cross-seed geometry deduplication and one global non-dominated sort.
Final Pareto artifacts are written only after all 528 authoritative tasks
finish successfully and exactly 168,960 terminal rows have been collected.
Results remain screening-only and are never promoted automatically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping
import urllib.parse
import urllib.request

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import slurm_nsga_offload as transport
from tools.mft_goal_fixed_primary_5t_collect import (
    CORE_TARGETS,
    WINDING_TARGETS,
    _atomic_json,
    _read_json,
    _sha,
    _sha_file,
    _truth,
    _validate_seal,
    _write_csv,
)


SCHEMA = "mft-goal-fixed-lm2mh-targeted-global-nds-v3"
PARETO_SCHEMA = "mft-goal-fixed-lm2mh-targeted-pareto-manifest-v3"
SMOKE_SCHEMA = "mft-goal-fixed-lm2mh-official5-projected-smoke-v1"
FIRST_TERMINAL_SMOKE_SCHEMA = (
    "mft-goal-fixed-lm2mh-first-terminal-smoke-v1"
)
LEGACY_CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-targeted-v1"
)
SPLITTEMP_CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-splittemp-v2"
)
ROLLING_CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-"
    "splittemp-v3-rolling512"
)
CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-"
    "old16-plus-splittemp512-global-v3"
)
RESULT_SCHEMA = "mft-goal-fixed-primary-5t-nsga-result-v1"
SUBMISSION_SCHEMA = "mft-goal-fixed-primary-5t-nsga-submission-v1"
FIXED_TERMINAL_SCHEMA = "mft-goal-fixed-lm2mh-terminal-candidates-v1"
PREOPT_SCHEMA = "mft-goal-fixed-primary-5t-pre-optimization-v1"
INITIALIZATION_SCHEMA = "mft-goal-fixed-lm2mh-targeted-initialization-v1"
LEGACY_HARD_SPEC_SHA256 = (
    "227cdc0db3b8dae490275d549e8aea93b295a75ee97ea98cdaa601bb96591298"
)
SPLITTEMP_HARD_SPEC_SHA256 = (
    "486418c731af63007915c9dd2034df1c65df80a5543546915aa25334c89d164f"
)
LEGACY_RUNNER_SOURCE_SHA256 = (
    "d82458cbc9793b199163653e0e74df99226d82d563f87811a13ec9f0cddd8673"
)
EXPECTED_RESONANCE_CONTRACT_SHA256 = (
    "9858b78f3084d344077fe0d90f7e390f7485d18248d1d711d2fde22c36c01ed1"
)
TASK_PREFIX = "mft-5t-lm2-"
REMOTE_BUNDLE = (
    "/gpfs/tmp_cpu2/mft_goal_20260726/"
    "mft-goal-763dbb46ae74e1722969d2c2"
)
RUNTIME_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
)
LEGACY_SEEDS = tuple(range(2_707_276_100, 2_707_276_116))
SPLITTEMP_V2_SEEDS = tuple(range(2_707_277_000, 2_707_277_256))
ROLLING_V3_SEEDS = tuple(range(2_707_277_256, 2_707_277_512))
FRESH_SEEDS = SPLITTEMP_V2_SEEDS + ROLLING_V3_SEEDS
SOURCE_SPECS = (
    {
        "label": "legacy-v1-seeds16",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_targeted_w1200_l1000_v1"
        / "submission_manifest.json",
        "payload_sha256": (
            "2189ca3d0c2db606c014b5a33260af8ae6f1af5cb6ad9b2920cda47022e15604"
        ),
        "campaign_id": LEGACY_CAMPAIGN_ID,
        "hard_spec_sha256": LEGACY_HARD_SPEC_SHA256,
        "runner_source_sha256": LEGACY_RUNNER_SOURCE_SHA256,
        "seeds": LEGACY_SEEDS,
        "task_ids": tuple(range(96_416, 96_432)),
        "campaign_total_seed_count": 16,
        "temperature_constraint_mode": "legacy_100C_reclassify_to_splittemp",
        "replacement": False,
    },
    {
        "label": "splittemp-v2-seeds0000-0063",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v2_fresh64"
        / "submission_manifest.json",
        "payload_sha256": (
            "519839038d3b88bff1e1b896c58dbedc60250ce336830ec441319375e1fa34c5"
        ),
        "campaign_id": SPLITTEMP_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "58a4f1aaf3a480012eb062a0eb64630d8af9ebe7aea93430ef406e3662c9fd8b"
        ),
        "seeds": tuple(range(2_707_277_000, 2_707_277_064)),
        "task_ids": tuple(range(96_485, 96_549)),
        "campaign_total_seed_count": 64,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v2-seeds0064-0127",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v2_seeds0064_0127"
        / "submission_manifest.json",
        "payload_sha256": (
            "86ded437c9282f3bdaba62bca186e71134b819a61dab5a2a9632fcb220fd6766"
        ),
        "campaign_id": SPLITTEMP_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "d40b3de4d33fc18763d5ac3085007de96de3e260f6db1cc56c288e28e71f7cdd"
        ),
        "seeds": tuple(range(2_707_277_064, 2_707_277_128)),
        "task_ids": tuple(range(96_549, 96_613)),
        "campaign_total_seed_count": 256,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v2-seeds0128-0191",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v2_seeds0128_0191"
        / "submission_manifest.json",
        "payload_sha256": (
            "1fcecd3f0fa73b65771837127b711c52c941b097d984715a15ab43d1054543f5"
        ),
        "campaign_id": SPLITTEMP_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "d40b3de4d33fc18763d5ac3085007de96de3e260f6db1cc56c288e28e71f7cdd"
        ),
        "seeds": tuple(range(2_707_277_128, 2_707_277_192)),
        "task_ids": tuple(range(96_613, 96_677)),
        "campaign_total_seed_count": 256,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v2-seeds0192-0255",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v2_seeds0192_0255"
        / "submission_manifest.json",
        "payload_sha256": (
            "0ff95875035855ba3b763c8c2185e86d91aed3365b49f082348b77d3175ed24b"
        ),
        "campaign_id": SPLITTEMP_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "d40b3de4d33fc18763d5ac3085007de96de3e260f6db1cc56c288e28e71f7cdd"
        ),
        "seeds": tuple(range(2_707_277_192, 2_707_277_256)),
        "task_ids": tuple(range(96_677, 96_741)),
        "campaign_total_seed_count": 256,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v3-seeds0256-0319",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v3_seeds0256_0319"
        / "submission_manifest.json",
        "payload_sha256": (
            "c47dbb3ad665682f5b16e1a152f6c680498ddf31b1dac1cee6d9c7c1b09b679a"
        ),
        "campaign_id": ROLLING_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "3a5d2615b8e69a2caff0c6c56986a7b9f3ee1bf1ffec09d1e6ef752ae86ef91f"
        ),
        "seeds": tuple(range(2_707_277_256, 2_707_277_320)),
        "task_ids": tuple(range(96_756, 96_820)),
        "campaign_total_seed_count": 512,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v3-seeds0320-0383",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v3_seeds0320_0383"
        / "submission_manifest.json",
        "payload_sha256": (
            "30861bd4b284562c7601194a04c7390d58c12637d644d14a100b7e77a96a2d08"
        ),
        "campaign_id": ROLLING_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "3a5d2615b8e69a2caff0c6c56986a7b9f3ee1bf1ffec09d1e6ef752ae86ef91f"
        ),
        "seeds": tuple(range(2_707_277_320, 2_707_277_384)),
        "task_ids": tuple(range(96_820, 96_884)),
        "campaign_total_seed_count": 512,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v3-seeds0384-0447",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v3_seeds0384_0447"
        / "submission_manifest.json",
        "payload_sha256": (
            "f0891a9d44a80f90ad84d602a310b3e078d678ef8a8261a3720ecab7d898332d"
        ),
        "campaign_id": ROLLING_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "3a5d2615b8e69a2caff0c6c56986a7b9f3ee1bf1ffec09d1e6ef752ae86ef91f"
        ),
        "seeds": tuple(range(2_707_277_384, 2_707_277_448)),
        "task_ids": tuple(range(96_884, 96_948)),
        "campaign_total_seed_count": 512,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v3-seeds0448-0511",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v3_seeds0448_0511"
        / "submission_manifest.json",
        "payload_sha256": (
            "6cda7e26417d0f1d3a4ba556b8cfbfa583e1a5d852b5f2e87ebf794642631a4f"
        ),
        "campaign_id": ROLLING_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "3a5d2615b8e69a2caff0c6c56986a7b9f3ee1bf1ffec09d1e6ef752ae86ef91f"
        ),
        "seeds": tuple(range(2_707_277_448, 2_707_277_512)),
        "task_ids": tuple(range(96_948, 97_012)),
        "campaign_total_seed_count": 512,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": False,
    },
    {
        "label": "splittemp-v2-retry-seed0022",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v2_retry_seed0022"
        / "submission_manifest.json",
        "payload_sha256": (
            "5359656679b032249d442cfde7a71ca8eff2f7d7339426cce01b17f37704de56"
        ),
        "campaign_id": SPLITTEMP_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "563a02ea0540aa34a73bf77df9ccf7b564add15430e32204769ccfdb9dae2568"
        ),
        "seeds": (2_707_277_022,),
        "task_ids": (96_741,),
        "campaign_total_seed_count": 256,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": True,
    },
    {
        "label": "splittemp-v2-retry-seed0061",
        "path": RUNTIME_ROOT
        / "fixed_lm2mh_splittemp_v2_retry_seed0061"
        / "submission_manifest.json",
        "payload_sha256": (
            "301e0a37e51956a83ceb26ee8add394f3a4d3976147deb9886688dbafaebefce"
        ),
        "campaign_id": SPLITTEMP_CAMPAIGN_ID,
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": (
            "563a02ea0540aa34a73bf77df9ccf7b564add15430e32204769ccfdb9dae2568"
        ),
        "seeds": (2_707_277_061,),
        "task_ids": (96_742,),
        "campaign_total_seed_count": 256,
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": True,
    },
)
BASE_MANIFESTS = tuple(
    Path(spec["path"]) for spec in SOURCE_SPECS if not spec["replacement"]
)
RETRY_SEARCH_ROOTS = (
    RUNTIME_ROOT,
    REPOSITORY_ROOT / "artifacts" / "mft_goal_20260726",
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_old16_plus_splittemp512_global_nds_v3"
)
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
SCHEDULER_URL = "http://127.0.0.1:8002"
TERMINAL_ROWS_PER_TASK = 320
EXPECTED_SEED_COUNT = len(LEGACY_SEEDS) + len(FRESH_SEEDS)
EXPECTED_RAW_ROWS = EXPECTED_SEED_COUNT * TERMINAL_ROWS_PER_TASK
BASE_TASK_BY_SEED = {
    **{
        seed: 96_485 + seed - SPLITTEMP_V2_SEEDS[0]
        for seed in SPLITTEMP_V2_SEEDS
    },
    **{
        seed: 96_756 + seed - ROLLING_V3_SEEDS[0]
        for seed in ROLLING_V3_SEEDS
    },
}
FINAL_FILES = (
    "global_terminal_all_seeds_raw.csv",
    "global_terminal_candidates.csv",
    "global_pareto_front.csv",
    "global_conditional_nonthermal_pareto_front.csv",
    "global_minimum_violation_objective_front.csv",
    "fea_acquisition_candidates.csv",
)


def _api_any(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _discover_manifest_paths() -> tuple[Path, ...]:
    paths = {path.resolve(strict=True) for path in BASE_MANIFESTS}
    for root in RETRY_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.glob(
            "fixed_lm2mh_splittemp_v*_retry_seed*/submission_manifest.json"
        ):
            paths.add(path.resolve(strict=True))
    return tuple(sorted(paths, key=str))


def _source_spec(
    path: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    matches = [
        spec
        for spec in SOURCE_SPECS
        if Path(spec["path"]).resolve(strict=True) == resolved
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RuntimeError(f"ambiguous aggregate source: {resolved}")
    allowed_root = any(
        resolved.is_relative_to(root.resolve())
        for root in RETRY_SEARCH_ROOTS
        if root.exists()
    )
    match = re.fullmatch(
        r"fixed_lm2mh_splittemp_v[23]_retry_seed(s?)"
        r"(\d{4}(?:_\d{4})*)",
        resolved.parent.name,
    )
    entries = list(manifest.get("submissions") or [])
    declared_offsets = (
        tuple(int(value) for value in match.group(2).split("_"))
        if match is not None
        else ()
    )
    if (
        not allowed_root
        or match is None
        or not entries
        or len(entries) != len(declared_offsets)
        or (len(entries) == 1) != (match.group(1) == "")
        or manifest.get("campaign_id")
        not in {SPLITTEMP_CAMPAIGN_ID, ROLLING_CAMPAIGN_ID}
        or manifest.get("hard_spec_sha256")
        != SPLITTEMP_HARD_SPEC_SHA256
    ):
        raise RuntimeError(
            f"manifest is not an exact aggregate source: {resolved}"
        )
    seeds = tuple(sorted(int(entry["seed"]) for entry in entries))
    task_ids = tuple(sorted(int(entry["task_id"]) for entry in entries))
    offsets = tuple(sorted(seed - FRESH_SEEDS[0] for seed in seeds))
    runner = str(manifest.get("runner_source_sha256") or "").lower()
    if (
        len(set(seeds)) != len(seeds)
        or len(set(task_ids)) != len(task_ids)
        or any(seed not in set(FRESH_SEEDS) for seed in seeds)
        or tuple(sorted(declared_offsets)) != offsets
        or any(
            int(entry["task_id"])
            <= BASE_TASK_BY_SEED[int(entry["seed"])]
            for entry in entries
        )
        or not re.fullmatch(r"[0-9a-f]{64}", runner)
    ):
        raise RuntimeError(f"replacement manifest identity drifted: {resolved}")
    return {
        "label": (
            "replacement-"
            + "-".join(f"seed{offset:04d}" for offset in offsets)
            + "-"
            + "-".join(f"task{task_id}" for task_id in task_ids)
        ),
        "path": resolved,
        "payload_sha256": manifest["payload_sha256"],
        "campaign_id": manifest["campaign_id"],
        "hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": runner,
        "seeds": seeds,
        "task_ids": task_ids,
        "campaign_total_seed_count": int(
            manifest["campaign_total_seed_count"]
        ),
        "temperature_constraint_mode": "splittemp_native_100C_120C",
        "replacement": True,
    }


def _submission(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Mapping[str, Any]]:
    manifest = _validate_seal(
        _read_json(path.resolve(strict=True)), SUBMISSION_SCHEMA
    )
    spec = _source_spec(path, manifest)
    entries = list(manifest.get("submissions") or [])
    task_ids = tuple(sorted(int(item["task_id"]) for item in entries))
    seeds = tuple(sorted(int(item["seed"]) for item in entries))
    if (
        manifest.get("payload_sha256") != spec["payload_sha256"]
        or manifest.get("campaign_id") != spec["campaign_id"]
        or manifest.get("hard_spec_sha256")
        != spec["hard_spec_sha256"]
        or manifest.get("runner_source_sha256")
        != spec["runner_source_sha256"]
        or manifest.get("population") != TERMINAL_ROWS_PER_TASK
        or manifest.get("generations") != 80
        or manifest.get("campaign_total_seed_count")
        != spec["campaign_total_seed_count"]
        or manifest.get("seed_count") != len(spec["seeds"])
        or task_ids != tuple(sorted(spec["task_ids"]))
        or seeds != tuple(sorted(spec["seeds"]))
        or len(set(seeds)) != len(seeds)
    ):
        raise RuntimeError(
            f"targeted fixed-Lm source coverage drifted: {spec['label']}"
        )
    hard_spec = manifest.get("hard_spec") or {}
    if (
        hard_spec.get("primary_conductor_thickness_mm") != 5.0
        or hard_spec.get("primary_interturn_gap_mm") != 1.6
        or hard_spec.get("magnetizing_inductance_H") != 0.002
        or (hard_spec.get("size_limits_mm") or {})
        != {"H": 750.0, "L": 1000.0, "W": 1200.0}
        or (
            hard_spec.get("axis_contract") or {}
        ).get("rotation_or_axis_swap_allowed")
        is not False
        or (
            hard_spec.get("resonance_contract") or {}
        ).get("classification")
        != "screening-only"
    ):
        raise RuntimeError("targeted fixed-Lm hard-spec content drifted")
    if spec["campaign_id"] == LEGACY_CAMPAIGN_ID:
        if (
            hard_spec.get("winding_temperature_max_C") != 100.0
            or "temperature_family_limits_C" in hard_spec
        ):
            raise RuntimeError("legacy temperature contract drifted")
    elif (
        hard_spec.get("temperature_family_limits_C")
        != {
            "primary_winding": 100.0,
            "secondary_winding": 120.0,
            "core": 120.0,
        }
        or hard_spec.get(
            "legacy_scalar_winding_temperature_limit_allowed"
        )
        is not False
        or "winding_temperature_max_C" in hard_spec
    ):
        raise RuntimeError("split-temperature contract drifted")
    enriched = []
    for item in entries:
        entry = dict(item)
        entry["_collector_source_label"] = spec["label"]
        entry["_collector_campaign_id"] = spec["campaign_id"]
        entry["_collector_hard_spec_sha256"] = spec[
            "hard_spec_sha256"
        ]
        entry["_collector_runner_source_sha256"] = spec[
            "runner_source_sha256"
        ]
        entry["_collector_manifest_payload_sha256"] = manifest[
            "payload_sha256"
        ]
        entry["_collector_temperature_constraint_mode"] = spec[
            "temperature_constraint_mode"
        ]
        entry["_collector_replacement"] = bool(spec["replacement"])
        enriched.append(entry)
    enriched.sort(key=lambda item: int(item["task_id"]))
    return enriched, manifest, spec


def _select_authoritative_entries(
    source_entries: Iterable[Mapping[str, Any]],
    status_by_id: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one task per seed, preferring active then highest successful retry."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in source_entries:
        entry = dict(item)
        family = (
            "legacy"
            if entry["_collector_campaign_id"] == LEGACY_CAMPAIGN_ID
            else "fresh"
        )
        grouped.setdefault((family, int(entry["seed"])), []).append(
            entry
        )
    selected = []
    superseded = []
    for (family, seed), candidates in sorted(grouped.items()):
        candidates.sort(key=lambda entry: int(entry["task_id"]))
        base = [entry for entry in candidates if not entry["_collector_replacement"]]
        replacements = [
            entry for entry in candidates if entry["_collector_replacement"]
        ]
        if len(base) != 1 or (
            family == "legacy" and replacements
        ):
            raise RuntimeError(
                f"seed source baseline coverage drifted: {family}:{seed}"
            )
        for replacement in replacements:
            if (
                family != "fresh"
                or int(replacement["task_id"])
                <= BASE_TASK_BY_SEED[seed]
            ):
                raise RuntimeError(
                    f"replacement task is not newer than baseline: {seed}"
                )
        if not replacements:
            choice = base[0]
        elif status_by_id is None:
            choice = replacements[-1]
        else:
            active = [
                entry
                for entry in replacements
                if str(
                    status_by_id[int(entry["task_id"])].get("status")
                )
                not in {"completed", "failed", "cancelled"}
            ]
            completed = [
                entry
                for entry in replacements
                if str(
                    status_by_id[int(entry["task_id"])].get("status")
                )
                == "completed"
            ]
            if active:
                choice = active[-1]
            elif completed:
                choice = completed[-1]
            else:
                choice = replacements[-1]
        selected.append(choice)
        for prior in candidates:
            if int(prior["task_id"]) == int(choice["task_id"]):
                continue
            superseded.append(
                {
                    "seed_family": family,
                    "seed": seed,
                    "superseded_task_id": int(prior["task_id"]),
                    "selected_task_id": int(choice["task_id"]),
                    "selected_is_replacement": bool(
                        choice["_collector_replacement"]
                    ),
                }
            )
    return (
        sorted(selected, key=lambda entry: int(entry["task_id"])),
        sorted(
            superseded,
            key=lambda row: (row["seed"], row["superseded_task_id"]),
        ),
    )


def _validate_combined_coverage(
    entries: Iterable[Mapping[str, Any]],
    superseded: Iterable[Mapping[str, Any]],
) -> None:
    entries = list(entries)
    by_family: dict[str, set[int]] = {}
    for entry in entries:
        family = (
            "legacy"
            if entry["_collector_campaign_id"] == LEGACY_CAMPAIGN_ID
            else "fresh"
        )
        by_family.setdefault(family, set()).add(int(entry["seed"]))
    valid_fresh_tasks = all(
        int(entry["task_id"]) >= BASE_TASK_BY_SEED[int(entry["seed"])]
        for entry in entries
        if entry["_collector_campaign_id"] != LEGACY_CAMPAIGN_ID
    )
    if (
        by_family.get("legacy") != set(LEGACY_SEEDS)
        or by_family.get("fresh") != set(FRESH_SEEDS)
        or set(by_family) != {"legacy", "fresh"}
        or len(entries) != EXPECTED_SEED_COUNT
        or not valid_fresh_tasks
    ):
        raise RuntimeError("combined old16+splittemp512 coverage drifted")


def _submissions(
    paths: Iterable[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paths = tuple(Path(path) for path in paths)
    expected = {Path(path).resolve(strict=True) for path in BASE_MANIFESTS}
    observed = {path.resolve(strict=True) for path in paths}
    if not expected.issubset(observed) or len(paths) != len(observed):
        raise RuntimeError("all nine exact baseline manifests are required")
    all_entries = []
    sources = []
    for path in paths:
        entries, manifest, spec = _submission(path)
        all_entries.extend(entries)
        sources.append(
            {
                "label": spec["label"],
                "path": str(path.resolve(strict=True)),
                "campaign_id": spec["campaign_id"],
                "temperature_constraint_mode": spec[
                    "temperature_constraint_mode"
                ],
                "seed_count": len(spec["seeds"]),
                "manifest_payload_sha256": manifest["payload_sha256"],
                "hard_spec_sha256": manifest["hard_spec_sha256"],
                "runner_source_sha256": manifest[
                    "runner_source_sha256"
                ],
            }
        )
    return all_entries, sources, []


def _status_inventory(
    entries: Iterable[Mapping[str, Any]], scheduler_url: str
) -> list[dict[str, Any]]:
    entries = list(entries)
    expected_ids = {int(entry["task_id"]) for entry in entries}
    query = urllib.parse.urlencode(
        {
            "compact": "true",
            "limit": "1000",
            "name_prefix": TASK_PREFIX,
        }
    )
    payload = _api_any(
        scheduler_url.rstrip("/") + "/api/tasks?" + query
    )
    tasks = payload if isinstance(payload, list) else payload.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("Scheduler compact task inventory is invalid")
    by_id = {
        int(task["id"]): task
        for task in tasks
        if isinstance(task, dict)
        and int(task.get("id") or -1) in expected_ids
    }
    statuses = []
    for entry in entries:
        task_id = int(entry["task_id"])
        task = by_id.get(task_id)
        if task is None or task.get("name") != entry["name"]:
            raise RuntimeError(f"Scheduler task {task_id} is missing/drifted")
        statuses.append(task)
    return statuses


def _status_detail(
    entry: Mapping[str, Any], scheduler_url: str
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    url = (
        scheduler_url.rstrip("/")
        + f"/api/tasks/{task_id}?include_output=true&output_limit=30000"
    )
    value = _api_any(url)
    if not isinstance(value, dict):
        raise RuntimeError(f"Scheduler task {task_id} detail is invalid")
    if (
        int(value.get("id", value.get("task_id", -1))) != task_id
        or value.get("name") != entry["name"]
        or int(value.get("cpus") or 0) != 8
        or int(value.get("memory_mb") or 0) != 65_536
        or value.get("remote_cwd") != REMOTE_BUNDLE
    ):
        raise RuntimeError(f"Scheduler task {task_id} identity drifted")
    return value


def _terminal_details(
    entries: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    scheduler_url: str,
) -> dict[int, dict[str, Any]]:
    terminal = {
        int(status["id"])
        for status in statuses
        if status.get("status") in {"completed", "failed", "cancelled"}
    }
    selected = [
        entry for entry in entries if int(entry["task_id"]) in terminal
    ]
    if not selected:
        return {}
    workers = min(8, len(selected))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(
            executor.map(
                lambda entry: _status_detail(entry, scheduler_url),
                selected,
            )
        )
    return {int(value["id"]): value for value in values}


def _cached_or_download(
    connection: Any,
    *,
    remote: str,
    destination: Path,
    expected: Mapping[str, Any] | None,
    max_bytes: int,
) -> dict[str, Any]:
    normalized = None
    if expected is not None:
        normalized = {
            "sha256": str(expected["sha256"]),
            "bytes": int(
                expected.get("size_bytes", expected.get("bytes"))
            ),
        }
    if destination.is_file():
        observed = {
            "sha256": _sha_file(destination),
            "bytes": destination.stat().st_size,
        }
        if normalized is None or observed == normalized:
            return {
                **observed,
                "local_sha256": observed["sha256"],
                "remote_sha256": (
                    observed["sha256"]
                    if normalized is None
                    else normalized["sha256"]
                ),
                "remote_bytes": (
                    observed["bytes"]
                    if normalized is None
                    else normalized["bytes"]
                ),
                "verified": True,
                "transport": "authenticated_local_cache",
                "attempts": 0,
            }
        raise RuntimeError(f"cached artifact identity drifted: {destination}")
    return transport._download_verified_sftp(
        connection,
        remote,
        destination,
        retries=3,
        expected_identity=normalized,
        max_bytes=max_bytes,
    )


def _validate_result(
    entry: Mapping[str, Any],
    detail: Mapping[str, Any],
    *,
    authoritative_file_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    result = (
        dict(authoritative_file_result)
        if authoritative_file_result is not None
        else _result_from_detail(detail)
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"task {task_id} has no Scheduler RESULT_JSON")
    result = _validate_seal(result, RESULT_SCHEMA)
    if (
        result.get("campaign_id")
        != entry["_collector_campaign_id"]
        or result.get("lane") != "fixed-lm-targeted"
        or result.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or result.get("physics_sha256") != entry["physics_sha256"]
        or result.get("hard_spec_sha256")
        != entry["_collector_hard_spec_sha256"]
        or result.get("fixed_lm2mh_resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or result.get("terminal_population_count")
        != TERMINAL_ROWS_PER_TASK
        or result.get("terminal_primary_controls_attested") is not True
        or result.get("terminal_gap1_hard_fixed") is not True
        or result.get("terminal_primary_control_escape_count") != 0
        or result.get("global_nds_ready") is not True
        or result.get("screening_only") is not True
        or result.get("production_eligible") is not False
        or result.get(
            "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current"
        )
        is not False
    ):
        raise RuntimeError(f"task {task_id} result identity drifted")
    return result


def _result_from_detail(detail: Mapping[str, Any]) -> dict[str, Any] | None:
    """Use Scheduler RESULT_JSON, with its stdout copy as a sealed fallback."""

    direct = detail.get("result_json")
    if isinstance(direct, dict):
        return direct
    stdout = detail.get("stdout")
    if isinstance(stdout, list):
        stdout = "\n".join(str(value) for value in stdout)
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith("RESULT_JSON="):
            try:
                value = json.loads(line[len("RESULT_JSON=") :])
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None
    return None


def _download_task(
    *,
    connection: Any,
    entry: Mapping[str, Any],
    detail: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    task_root = output / f"task-{task_id}"
    remote_root = (
        f"{REMOTE_BUNDLE}/runs/fixed-lm2mh-targeted-task-{task_id}"
    )
    records = {
        "result.json": _cached_or_download(
            connection,
            remote=f"{remote_root}/result.json",
            destination=task_root / "result.json",
            expected=None,
            max_bytes=16 * 1024 * 1024,
        )
    }
    local_result = _validate_seal(
        _read_json(task_root / "result.json"), RESULT_SCHEMA
    )
    scheduler_result = _result_from_detail(detail)
    if scheduler_result is not None and local_result != scheduler_result:
        raise RuntimeError(f"task {task_id} RESULT_JSON/file mismatch")
    result = _validate_result(
        entry, detail, authoritative_file_result=local_result
    )
    inventory = result.get("artifact_inventory") or {}
    names = (
        "terminal_fixed_lm2mh_candidates.csv",
        "terminal_fixed_lm2mh_candidates.manifest.json",
    )
    if any(name not in inventory for name in names):
        raise RuntimeError(f"task {task_id} fixed-Lm inventory is incomplete")
    for name in names:
        records[name] = _cached_or_download(
            connection,
            remote=f"{remote_root}/{name}",
            destination=task_root / name,
            expected=inventory[name],
            max_bytes=128 * 1024 * 1024,
        )
    fixed_manifest = _validate_seal(
        _read_json(
            task_root
            / "terminal_fixed_lm2mh_candidates.manifest.json"
        ),
        FIXED_TERMINAL_SCHEMA,
    )
    csv_path = task_root / "terminal_fixed_lm2mh_candidates.csv"
    if (
        fixed_manifest.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or fixed_manifest.get("row_count") != TERMINAL_ROWS_PER_TASK
        or fixed_manifest.get("resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or fixed_manifest.get("csv_sha256") != _sha_file(csv_path)
        or fixed_manifest.get("thermal_screening_only") is not True
        or fixed_manifest.get("production_eligible") is not False
    ):
        raise RuntimeError(f"task {task_id} fixed terminal drifted")
    return {
        "task_id": task_id,
        "seed": int(entry["seed"]),
        "fixed_primary_turns": int(entry["fixed_primary_turns"]),
        "source_label": entry["_collector_source_label"],
        "source_campaign_id": entry["_collector_campaign_id"],
        "source_hard_spec_sha256": entry[
            "_collector_hard_spec_sha256"
        ],
        "source_runner_source_sha256": entry[
            "_collector_runner_source_sha256"
        ],
        "source_manifest_payload_sha256": entry[
            "_collector_manifest_payload_sha256"
        ],
        "temperature_constraint_mode": entry[
            "_collector_temperature_constraint_mode"
        ],
        "replacement_task": bool(entry["_collector_replacement"]),
        "result_payload_sha256": result["payload_sha256"],
        "fixed_terminal_manifest_payload_sha256": fixed_manifest[
            "payload_sha256"
        ],
        "screening_feasible_count": int(
            fixed_manifest["screening_feasible_count"]
        ),
        "csv": str(csv_path),
        "csv_sha256": _sha_file(csv_path),
        "download_records": records,
    }


def _official5_smoke(
    *,
    connection: Any,
    entry: Mapping[str, Any],
    status: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    local = output / "smoke" / f"task-{task_id}" / "pre_optimization.json"
    remote = (
        f"{REMOTE_BUNDLE}/runs/fixed-lm2mh-targeted-task-{task_id}/"
        "pre_optimization.json"
    )
    try:
        record = _cached_or_download(
            connection,
            remote=remote,
            destination=local,
            expected=None,
            max_bytes=16 * 1024 * 1024,
        )
    except Exception as exc:
        if status.get("status") not in {"completed", "failed", "cancelled"}:
            return {
                "verified": False,
                "state": "worker_preoptimization_pending",
                "task_id": task_id,
                "detail": f"{type(exc).__name__}:{exc}",
            }
        raise
    preopt = _validate_seal(_read_json(local), PREOPT_SCHEMA)
    initialization = _validate_seal(
        dict(preopt.get("initialization") or {}), INITIALIZATION_SCHEMA
    )
    smoke = initialization.get("projected_anchor_smoke") or {}
    physical_g = smoke.get("physical_G") or {}
    required_g = (
        "exterior_width_limit",
        "exterior_length_limit",
        "exterior_height_limit",
        "half_magnetizing_resonance_minimum",
    )
    if (
        preopt.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or preopt.get("hard_spec_sha256")
        != entry["_collector_hard_spec_sha256"]
        or preopt.get("fixed_lm2mh_resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or preopt.get("screening_only") is not True
        or (preopt.get("axis_contract") or {}).get(
            "rotation_or_axis_swap_allowed"
        )
        is not False
        or initialization.get("official5_projected_anchor_included")
        is not True
        or initialization.get("official5_projected_anchor_first_row")
        is not True
        or float(smoke.get("cw1_mm", math.nan)) != 5.0
        or float(smoke.get("gap1_mm", math.nan)) != 1.6
        or int(smoke.get("N1", -1)) != 6
        or any(name not in physical_g for name in required_g)
    ):
        raise RuntimeError("official#5 projected worker smoke drifted")
    observed = {
        "W_drawing_x_mm": 1200.0
        + float(physical_g["exterior_width_limit"]),
        "L_perpendicular_y_mm": 1000.0
        + float(physical_g["exterior_length_limit"]),
        "H_mm": 750.0 + float(physical_g["exterior_height_limit"]),
        "resonance_min_Hz_fixed_lm2mh": 15_000.0
        - float(physical_g["half_magnetizing_resonance_minimum"]),
    }
    if any(not math.isfinite(value) for value in observed.values()):
        raise RuntimeError("official#5 projected smoke is non-finite")
    result = {
        "schema_version": SMOKE_SCHEMA,
        "verified": True,
        "state": "verified_worker_preoptimization",
        "task_id": task_id,
        "seed": int(entry["seed"]),
        "official5_projected_anchor": True,
        "cw1_mm": 5.0,
        "gap1_mm": 1.6,
        "N1": 6,
        "axis_contract": {
            "W": "drawing_x_original_973mm_direction",
            "L": "drawing_y_perpendicular_direction",
            "rotation_or_axis_swap_allowed": False,
        },
        **observed,
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "hard_spec_sha256": entry["_collector_hard_spec_sha256"],
        "pre_optimization_sha256": _sha_file(local),
        "download_record": record,
        "screening_only": True,
        "production_eligible": False,
    }
    result["payload_sha256"] = _sha(result)
    _atomic_json(output / "official5_projected_smoke.json", result)
    return result


PRIMARY_WINDING_TARGETS = (
    "T_max_Tx",
    "Tprobe_Tx_leeward_max",
)
SECONDARY_WINDING_TARGETS = tuple(
    name for name in WINDING_TARGETS if name not in PRIMARY_WINDING_TARGETS
)


def _reclassify_secondary_temperature_constraints(
    physical_g: Mapping[str, Any],
    normalized_g: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert source 100 C winding constraints to the split 100/120 C gate."""

    physical = {name: float(value) for name, value in physical_g.items()}
    normalized = {name: float(value) for name, value in normalized_g.items()}
    for target in SECONDARY_WINDING_TARGETS:
        name = f"temperature_robust_limit:{target}"
        if name in physical:
            physical[name] -= 20.0
            # Every source targeted task used the sealed 10 C thermal scale.
            normalized[name] = physical[name] / 10.0
    return physical, normalized


def _aggregate_temperature_constraints(
    physical_g: Mapping[str, Any],
    normalized_g: Mapping[str, Any],
    mode: str,
) -> tuple[dict[str, float], dict[str, float]]:
    if mode == "legacy_100C_reclassify_to_splittemp":
        return _reclassify_secondary_temperature_constraints(
            physical_g, normalized_g
        )
    if mode == "splittemp_native_100C_120C":
        # V2 applied the 20 C secondary allowance inside each NSGA
        # evaluation before scaling and sorting.  Subtracting it again here
        # would silently turn the accepted secondary limit into 140 C.
        return (
            {name: float(value) for name, value in physical_g.items()},
            {name: float(value) for name, value in normalized_g.items()},
        )
    raise RuntimeError(f"unknown temperature constraint mode: {mode!r}")


def _temperatures(
    physical_g: Mapping[str, Any],
) -> tuple[float, float, float]:
    primary = [
        100.0 + float(physical_g[f"temperature_robust_limit:{name}"])
        for name in PRIMARY_WINDING_TARGETS
        if f"temperature_robust_limit:{name}" in physical_g
    ]
    secondary = [
        120.0 + float(physical_g[f"temperature_robust_limit:{name}"])
        for name in SECONDARY_WINDING_TARGETS
        if f"temperature_robust_limit:{name}" in physical_g
    ]
    core = [
        120.0 + float(physical_g[f"temperature_robust_limit:{name}"])
        for name in CORE_TARGETS
        if f"temperature_robust_limit:{name}" in physical_g
    ]
    if not primary or not secondary or not core:
        raise RuntimeError("terminal thermal constraints are incomplete")
    return max(primary), max(secondary), max(core)


def _load_rows(
    collections: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = []
    deduplicated: dict[str, dict[str, Any]] = {}
    for collection in collections:
        task_id = int(collection["task_id"])
        expected_turns = int(collection["fixed_primary_turns"])
        temperature_mode = str(
            collection["temperature_constraint_mode"]
        )
        with Path(collection["csv"]).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            required_columns = {
                "decoded_physical_params_json",
                "physical_geometry_sha256",
                "physical_G_fixed_lm2mh_json",
                "normalized_G_json",
                "screening_feasible_fixed_lm2mh",
                "fixed_lm2mh_resonance_contract_sha256",
                "fTx_Hz_fixed_lm2mh",
                "fRx_Hz_fixed_lm2mh",
                "fInter_Hz_fixed_lm2mh_diagnostic",
                "resonance_min_Hz_fixed_lm2mh",
                "objective_volume_L",
                "objective_total_loss_W",
            }
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise RuntimeError(
                    f"task {task_id} fixed terminal columns drifted"
                )
            task_rows = 0
            for csv_row_index, row in enumerate(reader, start=2):
                task_rows += 1
                params = json.loads(row["decoded_physical_params_json"])
                source_physical_g = json.loads(
                    row["physical_G_fixed_lm2mh_json"]
                )
                source_normalized_g = json.loads(row["normalized_G_json"])
                f_tx = float(row["fTx_Hz_fixed_lm2mh"])
                f_rx = float(row["fRx_Hz_fixed_lm2mh"])
                f_inter = float(
                    row["fInter_Hz_fixed_lm2mh_diagnostic"]
                )
                f_min = float(row["resonance_min_Hz_fixed_lm2mh"])
                source_expected_feasible = (
                    _truth(row["decoder_valid"])
                    and _truth(row["surrogate_physical_valid"])
                    and all(
                        float(value) <= 0.0
                        for value in source_physical_g.values()
                    )
                )
                if (
                    int(params["N1"]) != expected_turns
                    or float(params["cw1"]) != 5.0
                    or float(params["gap1"]) != 1.6
                    or row["fixed_lm2mh_resonance_contract_sha256"]
                    != EXPECTED_RESONANCE_CONTRACT_SHA256
                    or not all(
                        math.isfinite(value) and value > 0.0
                        for value in (f_tx, f_rx, f_inter, f_min)
                    )
                    or not math.isclose(
                        f_min,
                        min(f_tx, f_rx),
                        rel_tol=1e-12,
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        float(
                            source_physical_g[
                                "fixed_Lm2mH_self_resonance_minimum"
                            ]
                        ),
                        15_000.0 - f_min,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    or _truth(
                        row["screening_feasible_fixed_lm2mh"]
                    )
                    != source_expected_feasible
                    or not _truth(row["screening_only_fixed_lm2mh"])
                    or _truth(row["production_eligible_fixed_lm2mh"])
                ):
                    raise RuntimeError(
                        "fixed-Lm terminal row contract drifted: "
                        f"task={task_id} csv_row={csv_row_index}"
                    )
                physical_g, normalized_g = (
                    _aggregate_temperature_constraints(
                        source_physical_g,
                        source_normalized_g,
                        temperature_mode,
                    )
                )
                expected_feasible = (
                    _truth(row["decoder_valid"])
                    and _truth(row["surrogate_physical_valid"])
                    and all(
                        float(value) <= 0.0
                        for value in physical_g.values()
                    )
                )
                primary_max, secondary_max, core_max = _temperatures(
                    physical_g
                )
                violation = math.sqrt(
                    sum(
                        max(float(value), 0.0) ** 2
                        for value in normalized_g.values()
                    )
                )
                conditional_violation = math.sqrt(
                    sum(
                        max(float(value), 0.0) ** 2
                        for name, value in normalized_g.items()
                        if not name.startswith("temperature_robust_limit:")
                    )
                )
                conditional = (
                    _truth(row["decoder_valid"])
                    and _truth(row["surrogate_physical_valid"])
                    and all(
                        float(value) <= 0.0
                        for name, value in physical_g.items()
                        if not name.startswith("temperature_robust_limit:")
                    )
                )
                enriched = {
                    **row,
                    "scheduler_task_id": task_id,
                    "source_campaign_id": collection[
                        "source_campaign_id"
                    ],
                    "source_manifest_payload_sha256": collection[
                        "source_manifest_payload_sha256"
                    ],
                    "source_temperature_constraint_mode": (
                        temperature_mode
                    ),
                    "source_replacement_task": bool(
                        collection["replacement_task"]
                    ),
                    "fixed_primary_turns_stratum": expected_turns,
                    "exterior_W_drawing_x_mm": 1200.0
                    + float(physical_g["exterior_width_limit"]),
                    "exterior_L_perpendicular_y_mm": 1000.0
                    + float(physical_g["exterior_length_limit"]),
                    "exterior_H_mm": 750.0
                    + float(physical_g["exterior_height_limit"]),
                    "primary_winding_robust_max_C_screening": primary_max,
                    "secondary_winding_robust_max_C_screening": secondary_max,
                    "winding_robust_max_C_screening": max(
                        primary_max, secondary_max
                    ),
                    "core_robust_max_C_screening": core_max,
                    "physical_G_splittemp_v2_json": json.dumps(
                        physical_g,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "normalized_G_splittemp_v2_json": json.dumps(
                        normalized_g,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "normalized_constraint_violation_l2": violation,
                    "normalized_nonthermal_violation_l2": (
                        conditional_violation
                    ),
                    "conditional_nonthermal_feasible_diagnostic": (
                        conditional
                    ),
                    "global_screening_feasible": expected_feasible,
                    "global_production_eligible": False,
                }
                if any(
                    not math.isfinite(float(enriched[name]))
                    for name in (
                        "exterior_W_drawing_x_mm",
                        "exterior_L_perpendicular_y_mm",
                        "exterior_H_mm",
                        "primary_winding_robust_max_C_screening",
                        "secondary_winding_robust_max_C_screening",
                        "core_robust_max_C_screening",
                        "normalized_constraint_violation_l2",
                    )
                ):
                    raise RuntimeError(
                        f"task {task_id} terminal row is non-finite"
                    )
                raw_rows.append(enriched)
                identity = row["physical_geometry_sha256"]
                prior = deduplicated.get(identity)
                rank = (
                    violation,
                    float(row["objective_volume_L"]),
                    float(row["objective_total_loss_W"]),
                    task_id,
                )
                if prior is None or rank < (
                    float(prior["normalized_constraint_violation_l2"]),
                    float(prior["objective_volume_L"]),
                    float(prior["objective_total_loss_W"]),
                    int(prior["scheduler_task_id"]),
                ):
                    deduplicated[identity] = enriched
            if task_rows != TERMINAL_ROWS_PER_TASK:
                raise RuntimeError(
                    f"task {task_id} has {task_rows} terminal rows, expected 320"
                )
    raw_rows.sort(
        key=lambda row: (
            int(row["scheduler_task_id"]),
            int(row["terminal_population_index"]),
        )
    )
    unique_rows = sorted(
        deduplicated.values(),
        key=lambda row: (
            float(row["normalized_constraint_violation_l2"]),
            float(row["objective_volume_L"]),
            float(row["objective_total_loss_W"]),
            row["physical_geometry_sha256"],
        ),
    )
    return raw_rows, unique_rows


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    keys: tuple[str, ...],
) -> bool:
    observed = [
        (float(left[key]), float(right[key])) for key in keys
    ]
    return all(a <= b + 1e-12 for a, b in observed) and any(
        a < b - 1e-12 for a, b in observed
    )


def _non_dominated(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            *(float(row[key]) for key in keys),
            row["physical_geometry_sha256"],
        ),
    )
    front: list[dict[str, Any]] = []
    for row in ordered:
        if any(_dominates(prior, row, keys) for prior in front):
            continue
        front = [
            prior
            for prior in front
            if not _dominates(row, prior, keys)
        ]
        front.append(row)
    return sorted(
        front,
        key=lambda row: (
            float(row["objective_volume_L"]),
            float(row["objective_total_loss_W"]),
            row["physical_geometry_sha256"],
        ),
    )


def _acquisition_candidates(
    rows: list[dict[str, Any]],
    feasible_front: list[dict[str, Any]],
    conditional_front: list[dict[str, Any]],
    constraint_front: list[dict[str, Any]],
    count: int = 12,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    pool_map: dict[str, dict[str, Any]] = {}
    for row in (
        feasible_front
        + conditional_front
        + constraint_front
        + sorted(
            rows,
            key=lambda item: (
                float(item["normalized_constraint_violation_l2"]),
                float(item["objective_volume_L"]),
            ),
        )[:256]
    ):
        pool_map[row["physical_geometry_sha256"]] = row
    pool = list(pool_map.values())
    selected: list[tuple[str, dict[str, Any]]] = []
    seen = set()

    def add(role: str, row: dict[str, Any] | None) -> None:
        if row is None:
            return
        identity = row["physical_geometry_sha256"]
        if identity not in seen and len(selected) < count:
            selected.append((role, row))
            seen.add(identity)

    add(
        "least_normalized_constraint_violation",
        min(
            rows,
            key=lambda row: float(
                row["normalized_constraint_violation_l2"]
            ),
        ),
    )
    add(
        "minimum_volume",
        min(pool, key=lambda row: float(row["objective_volume_L"])),
    )
    add(
        "minimum_loss",
        min(pool, key=lambda row: float(row["objective_total_loss_W"])),
    )
    add(
        "maximum_fixed_lm2mh_resonance_margin",
        max(
            pool,
            key=lambda row: float(
                row["resonance_min_Hz_fixed_lm2mh"]
            ),
        ),
    )
    for column, role in (
        ("pred_C_tx_tx_F_fixed_lm2mh", "minimum_predicted_C_tx_tx"),
        ("pred_C_rx_rx_F_fixed_lm2mh", "minimum_predicted_C_rx_rx"),
        ("pred_C_tx_rx_F_fixed_lm2mh", "minimum_predicted_C_tx_rx"),
        (
            "primary_winding_robust_max_C_screening",
            "minimum_screened_primary_winding_temperature",
        ),
        (
            "secondary_winding_robust_max_C_screening",
            "minimum_screened_secondary_winding_temperature",
        ),
        (
            "core_robust_max_C_screening",
            "minimum_screened_core_temperature",
        ),
    ):
        add(role, min(pool, key=lambda row: float(row[column])))
    official_pool = [
        row
        for row in rows
        if int(row["fixed_primary_turns_stratum"]) == 6
    ]
    add(
        "nearest_official5_drawing_baseline_N1_6",
        (
            min(
                official_pool,
                key=lambda row: (
                    (
                        (
                            float(row["exterior_W_drawing_x_mm"])
                            - 1194.38
                        )
                        / 1200.0
                    )
                    ** 2
                    + (
                        (
                            float(
                                row[
                                    "exterior_L_perpendicular_y_mm"
                                ]
                            )
                            - 961.4
                        )
                        / 1000.0
                    )
                    ** 2
                    + (
                        (float(row["exterior_H_mm"]) - 707.0) / 750.0
                    )
                    ** 2
                ),
            )
            if official_pool
            else None
        ),
    )
    for turns in (5, 6, 7, 8):
        stratum = [
            row
            for row in rows
            if int(row["fixed_primary_turns_stratum"]) == turns
        ]
        add(
            f"N1_{turns}_least_violation",
            (
                min(
                    stratum,
                    key=lambda row: (
                        float(
                            row[
                                "normalized_constraint_violation_l2"
                            ]
                        ),
                        float(row["objective_volume_L"]),
                    ),
                )
                if stratum
                else None
            ),
        )

    dimensions = (
        "normalized_constraint_violation_l2",
        "objective_volume_L",
        "objective_total_loss_W",
        "resonance_min_Hz_fixed_lm2mh",
        "pred_C_tx_tx_F_fixed_lm2mh",
        "pred_C_rx_rx_F_fixed_lm2mh",
        "pred_C_tx_rx_F_fixed_lm2mh",
        "exterior_W_drawing_x_mm",
        "exterior_L_perpendicular_y_mm",
        "exterior_H_mm",
    )
    minima = {
        key: min(float(row[key]) for row in pool) for key in dimensions
    }
    spans = {
        key: max(float(row[key]) for row in pool) - minima[key]
        for key in dimensions
    }

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple(
            0.0
            if spans[key] <= 0.0
            else (float(row[key]) - minima[key]) / spans[key]
            for key in dimensions
        )

    def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        return math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(vector(left), vector(right))
            )
        )

    while len(selected) < min(count, len(pool)):
        remaining = [
            row
            for row in pool
            if row["physical_geometry_sha256"] not in seen
        ]
        if not remaining:
            break
        choice = max(
            remaining,
            key=lambda row: min(
                distance(row, prior) for _, prior in selected
            ),
        )
        add("farthest_point_diversity_fill", choice)
    return [
        {
            **row,
            "acquisition_rank": rank,
            "acquisition_selection_role": role,
            "acquisition_topology": "symmetric_unrounded",
            "acquisition_rounded_FEA_allowed": False,
            "acquisition_classification": (
                "screening-only-requires-symmetric-FEA"
            ),
            "acquisition_production_eligible": False,
        }
        for rank, (role, row) in enumerate(selected, start=1)
    ]


def _first_terminal_smoke(
    collection: Mapping[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    task_id = int(collection["task_id"])
    subset = [
        row for row in rows if int(row["scheduler_task_id"]) == task_id
    ]
    if len(subset) != TERMINAL_ROWS_PER_TASK:
        raise RuntimeError("first terminal smoke row coverage drifted")
    value = {
        "schema_version": FIRST_TERMINAL_SMOKE_SCHEMA,
        "verified": True,
        "task_id": task_id,
        "seed": int(collection["seed"]),
        "row_count": len(subset),
        "cw1_unique_mm": sorted(
            {float(json.loads(row["decoded_physical_params_json"])["cw1"]) for row in subset}
        ),
        "gap1_unique_mm": sorted(
            {float(json.loads(row["decoded_physical_params_json"])["gap1"]) for row in subset}
        ),
        "axis_contract": {
            "W": "drawing_x_original_973mm_direction",
            "L": "drawing_y_perpendicular_direction",
            "rotation_or_axis_swap_allowed": False,
        },
        "W_range_mm": [
            min(float(row["exterior_W_drawing_x_mm"]) for row in subset),
            max(float(row["exterior_W_drawing_x_mm"]) for row in subset),
        ],
        "L_range_mm": [
            min(
                float(row["exterior_L_perpendicular_y_mm"])
                for row in subset
            ),
            max(
                float(row["exterior_L_perpendicular_y_mm"])
                for row in subset
            ),
        ],
        "H_range_mm": [
            min(float(row["exterior_H_mm"]) for row in subset),
            max(float(row["exterior_H_mm"]) for row in subset),
        ],
        "fixed_lm2mh_resonance_min_range_Hz": [
            min(
                float(row["resonance_min_Hz_fixed_lm2mh"])
                for row in subset
            ),
            max(
                float(row["resonance_min_Hz_fixed_lm2mh"])
                for row in subset
            ),
        ],
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "screening_feasible_count": sum(
            bool(row["global_screening_feasible"]) for row in subset
        ),
        "screening_only": True,
        "production_eligible": False,
    }
    if (
        value["cw1_unique_mm"] != [5.0]
        or value["gap1_unique_mm"] != [1.6]
    ):
        raise RuntimeError("first terminal smoke fixed controls drifted")
    value["payload_sha256"] = _sha(value)
    _atomic_json(output / "first_completed_terminal_smoke.json", value)
    return value


def _write_partial(
    output: Path,
    raw_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    feasible_front: list[dict[str, Any]],
    conditional_front: list[dict[str, Any]],
    constraint_front: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
) -> None:
    _write_csv(output / "partial_global_terminal_all_seeds_raw.csv", raw_rows)
    _write_csv(output / "partial_global_terminal_candidates.csv", rows)
    _write_csv(
        output / "partial_global_screening_pareto_front.csv",
        feasible_front,
    )
    _write_csv(
        output / "partial_global_conditional_nonthermal_pareto_front.csv",
        conditional_front,
    )
    _write_csv(
        output / "partial_global_constraint_objective_front.csv",
        constraint_front,
    )
    _write_csv(
        output / "partial_fea_acquisition_candidates.csv",
        acquisition,
    )


def _write_final(
    output: Path,
    raw_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    feasible_front: list[dict[str, Any]],
    conditional_front: list[dict[str, Any]],
    constraint_front: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    superseded: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = {
        "global_terminal_all_seeds_raw.csv": raw_rows,
        "global_terminal_candidates.csv": rows,
        "global_pareto_front.csv": feasible_front,
        "global_conditional_nonthermal_pareto_front.csv": (
            conditional_front
        ),
        "global_minimum_violation_objective_front.csv": constraint_front,
        "fea_acquisition_candidates.csv": acquisition,
    }
    for name, content in targets.items():
        _write_csv(output / name, content)
    manifest = {
        "schema_version": PARETO_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_seed_count": EXPECTED_SEED_COUNT,
        "legacy_source_seed_count": len(LEGACY_SEEDS),
        "splittemp_v2_source_seed_count": len(SPLITTEMP_V2_SEEDS),
        "rolling_v3_source_seed_count": len(ROLLING_V3_SEEDS),
        "fresh_source_seed_count": len(FRESH_SEEDS),
        "source_raw_terminal_row_count": len(raw_rows),
        "geometry_deduplicated_candidate_count": len(rows),
        "global_screening_feasible_count": sum(
            bool(row["global_screening_feasible"]) for row in rows
        ),
        "global_pareto_count": len(feasible_front),
        "conditional_nonthermal_pareto_count": len(conditional_front),
        "minimum_violation_objective_front_count": len(constraint_front),
        "fea_acquisition_candidate_count": len(acquisition),
        "global_non_dominated_sorting_complete": True,
        "pareto_definition": (
            "cross-seed geometry-deduplicated non-dominated sorting on "
            "(objective_volume_L, objective_total_loss_W) after every fixed-"
            "Lm2mH physical screening constraint is <=0, with primary "
            "winding <=100 C, secondary winding <=120 C, and core <=120 C"
        ),
        "conditional_front_definition": (
            "diagnostic-only non-dominated sorting after excluding thermal "
            "constraints; never a substitute for global_pareto_front.csv"
        ),
        "minimum_violation_front_definition": (
            "diagnostic-only 3-objective non-dominated sorting on normalized "
            "constraint violation L2, volume, and loss"
        ),
        "aggregate_hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "source_hard_spec_sha256": {
            "legacy_v1": LEGACY_HARD_SPEC_SHA256,
            "splittemp_v2": SPLITTEMP_HARD_SPEC_SHA256,
            "rolling_v3": SPLITTEMP_HARD_SPEC_SHA256,
        },
        "source_manifests": sources,
        "superseded_tasks": superseded,
        "temperature_constraint_modes": {
            "legacy_v1": "secondary constraints reclassified once from 100 C to 120 C",
            "splittemp_v2": "native 100 C primary / 120 C secondary; no collector relaxation",
            "rolling_v3": "native 100 C primary / 120 C secondary; no collector relaxation",
        },
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current": False,
        "temperature_limits_C": {
            "primary_winding": 100.0,
            "secondary_winding": 120.0,
            "core": 120.0,
        },
        "screening_only": True,
        "production_eligible": False,
        "files": {
            name: {
                "path": str(output / name),
                "sha256": _sha_file(output / name),
                "row_count": len(targets[name]),
            }
            for name in FINAL_FILES
        },
    }
    manifest["payload_sha256"] = _sha(manifest)
    _atomic_json(output / "global_pareto_manifest.json", manifest)
    return manifest


def collect(
    *,
    manifest_paths: Iterable[Path],
    output: Path,
    scheduler_url: str,
    accounts: Path,
    scheduler_source: Path,
) -> dict[str, Any]:
    candidate_entries, sources, _unused = _submissions(manifest_paths)
    candidate_statuses = _status_inventory(
        candidate_entries, scheduler_url
    )
    candidate_status_by_id = {
        int(status["id"]): status for status in candidate_statuses
    }
    entries, superseded = _select_authoritative_entries(
        candidate_entries, candidate_status_by_id
    )
    _validate_combined_coverage(entries, superseded)
    selected_ids = {int(entry["task_id"]) for entry in entries}
    statuses = [
        status
        for status in candidate_statuses
        if int(status["id"]) in selected_ids
    ]
    details = _terminal_details(entries, statuses, scheduler_url)
    by_id = {int(entry["task_id"]): entry for entry in entries}
    scheduler_success_details = [
        detail
        for detail in details.values()
        if detail.get("status") == "completed"
        and int(detail.get("exit_code") or 0) == 0
    ]
    completed_details = list(scheduler_success_details)
    artifact_pending_details = [
        detail
        for detail in scheduler_success_details
        if not isinstance(_result_from_detail(detail), dict)
    ]
    failed_details = [
        detail
        for detail in details.values()
        if detail.get("status") in {"failed", "cancelled"}
        or (
            detail.get("status") == "completed"
            and int(detail.get("exit_code") or 0) != 0
        )
    ]
    collections = []
    official_entry = next(
        entry for entry in entries if int(entry["fixed_primary_turns"]) == 6
    )
    official_detail = details.get(int(official_entry["task_id"]))
    if official_detail is None:
        raise RuntimeError("official projected-smoke task is not terminal")
    account_sessions = {}
    for account_name in {
        str(detail.get("account_name") or "").strip()
        for detail in completed_details
    }:
        if not account_name:
            raise RuntimeError("terminal task account identity is absent")
        account_sessions[account_name] = transport._account(
            accounts.resolve(strict=True),
            scheduler_source.resolve(strict=True),
            account_name,
        )
    official_account_name = str(
        official_detail["account_name"]
    ).strip()
    official_account, official_ssh_session = account_sessions[
        official_account_name
    ]
    connection = transport._PersistentAccountConnection(
        official_account, official_ssh_session
    )
    try:
        official_smoke = _official5_smoke(
            connection=connection,
            entry=official_entry,
            status=official_detail,
            output=output,
        )
    finally:
        connection.close()

    def download_completed(detail: Mapping[str, Any]) -> dict[str, Any]:
        account_name = str(detail.get("account_name") or "").strip()
        account, ssh_session = account_sessions[account_name]
        worker_connection = transport._PersistentAccountConnection(
            account, ssh_session
        )
        try:
            task_id = int(detail["id"])
            return _download_task(
                connection=worker_connection,
                entry=by_id[task_id],
                detail=detail,
                output=output / "collected",
            )
        finally:
            worker_connection.close()

    ordered_completed = sorted(
        completed_details, key=lambda value: int(value["id"])
    )
    if ordered_completed:
        with ThreadPoolExecutor(
            max_workers=min(8, len(ordered_completed))
        ) as executor:
            collections = list(
                executor.map(download_completed, ordered_completed)
            )
    raw_rows, rows = _load_rows(collections)
    feasible = [
        row for row in rows if bool(row["global_screening_feasible"])
    ]
    conditional = [
        row
        for row in rows
        if bool(row["conditional_nonthermal_feasible_diagnostic"])
    ]
    objective_keys = ("objective_volume_L", "objective_total_loss_W")
    feasible_front = _non_dominated(feasible, objective_keys)
    conditional_front = _non_dominated(conditional, objective_keys)
    constraint_front = _non_dominated(
        rows,
        (
            "normalized_constraint_violation_l2",
            "objective_volume_L",
            "objective_total_loss_W",
        ),
    )
    acquisition = _acquisition_candidates(
        rows,
        feasible_front,
        conditional_front,
        constraint_front,
        count=12,
    )
    _write_partial(
        output,
        raw_rows,
        rows,
        feasible_front,
        conditional_front,
        constraint_front,
        acquisition,
    )
    first_terminal_smoke = (
        _first_terminal_smoke(collections[0], raw_rows, output)
        if collections
        else {
            "verified": False,
            "state": "awaiting_first_successful_terminal_task",
        }
    )
    status_counts: dict[str, int] = {}
    for status in statuses:
        name = str(status.get("status") or "unknown")
        status_counts[name] = status_counts.get(name, 0) + 1
    final_ready = (
        len(collections) == EXPECTED_SEED_COUNT
        and not failed_details
        and len(raw_rows) == EXPECTED_RAW_ROWS
        and status_counts.get("completed", 0)
        == EXPECTED_SEED_COUNT
    )
    pareto_manifest = (
        _write_final(
            output,
            raw_rows,
            rows,
            feasible_front,
            conditional_front,
            constraint_front,
            acquisition,
            sources,
            superseded,
        )
        if final_ready
        else None
    )
    result = {
        "schema_version": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "submission_manifests": sources,
        "superseded_tasks": superseded,
        "aggregate_hard_spec_sha256": SPLITTEMP_HARD_SPEC_SHA256,
        "source_hard_spec_sha256": {
            "legacy_v1": LEGACY_HARD_SPEC_SHA256,
            "splittemp_v2": SPLITTEMP_HARD_SPEC_SHA256,
            "rolling_v3": SPLITTEMP_HARD_SPEC_SHA256,
        },
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "expected_seed_count": EXPECTED_SEED_COUNT,
        "expected_legacy_seed_count": len(LEGACY_SEEDS),
        "expected_splittemp_v2_seed_count": len(SPLITTEMP_V2_SEEDS),
        "expected_rolling_v3_seed_count": len(ROLLING_V3_SEEDS),
        "expected_fresh_seed_count": len(FRESH_SEEDS),
        "successful_terminal_seed_count": len(collections),
        "status_counts": dict(sorted(status_counts.items())),
        "raw_terminal_row_count": len(raw_rows),
        "expected_raw_terminal_row_count": EXPECTED_RAW_ROWS,
        "geometry_deduplicated_candidate_count": len(rows),
        "global_screening_feasible_count": len(feasible),
        "partial_screening_pareto_count": len(feasible_front),
        "partial_conditional_nonthermal_pareto_count": len(
            conditional_front
        ),
        "partial_minimum_violation_objective_front_count": len(
            constraint_front
        ),
        "fea_acquisition_candidate_count": len(acquisition),
        "global_nds_final": final_ready,
        "final_files_written": bool(pareto_manifest),
        "official5_projected_smoke": official_smoke,
        "first_completed_terminal_smoke": first_terminal_smoke,
        "failed_terminal_tasks": [
            {
                "task_id": int(detail["id"]),
                "status": detail.get("status"),
                "exit_code": detail.get("exit_code"),
                "failure_message": detail.get("failure_message"),
                "stderr_tail": str(detail.get("stderr") or "")[-4000:],
            }
            for detail in failed_details
        ],
        "successful_terminal_artifact_pending_tasks": [
            {
                "task_id": int(detail["id"]),
                "status": detail.get("status"),
                "exit_code": detail.get("exit_code"),
                "state": "awaiting_scheduler_RESULT_JSON_visibility",
            }
            for detail in artifact_pending_details
        ],
        "collections": collections,
        "task_status": [
            {
                "task_id": int(status["id"]),
                "name": status["name"],
                "status": status["status"],
                "started_at": status.get("started_at"),
            }
            for status in statuses
        ],
        "classification": "screening-only",
        "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current": False,
        "production_eligible": False,
        "pareto_manifest_payload_sha256": (
            None
            if pareto_manifest is None
            else pareto_manifest["payload_sha256"]
        ),
    }
    result["payload_sha256"] = _sha(result)
    _atomic_json(output / "collector_status.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        dest="manifests",
        type=Path,
        action="append",
        default=None,
        help=(
            "source submission manifest; repeat for all nine baselines and "
            "any retries (default discovers sealed old16+fresh512+retries)"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.poll_seconds < 10.0 or args.poll_seconds > 300.0:
        raise RuntimeError("poll-seconds must be within 10..300")
    while True:
        result = collect(
            manifest_paths=(
                _discover_manifest_paths()
                if args.manifests is None
                else tuple(args.manifests)
            ),
            output=args.output.resolve(),
            scheduler_url=args.scheduler_url,
            accounts=args.accounts,
            scheduler_source=args.scheduler_source,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if (
            not args.watch
            or result["global_nds_final"]
            or result["failed_terminal_tasks"]
        ):
            return (
                0
                if result["global_nds_final"] or not args.watch
                else 2
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
