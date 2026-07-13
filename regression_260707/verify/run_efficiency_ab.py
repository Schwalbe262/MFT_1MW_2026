"""Run reproducible baseline/variant arms for an MFT solver efficiency A/B.

This script only launches local subprocesses. It does not submit scheduler or
cluster work. A supervisor may run one arm per cluster task with ``--arm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "run_simulation_260706.py"


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} JSON must contain an object: {source}")
    return payload


def merge_params(
    baseline: Mapping[str, Any], overlay_payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return baseline, variant, and normalized overlay.

    A plain overlay object applies directly to the variant. The optional
    structured form accepts ``{"baseline": {...}, "variant": {...}}`` so a
    single experiment file can pin common solver controls explicitly.
    """
    if set(overlay_payload).issubset({"baseline", "variant"}) and any(
        key in overlay_payload for key in ("baseline", "variant")
    ):
        base_overlay = overlay_payload.get("baseline", {})
        variant_overlay = overlay_payload.get("variant", {})
        if not isinstance(base_overlay, dict) or not isinstance(variant_overlay, dict):
            raise TypeError("structured overlay baseline/variant values must be objects")
        normalized = {
            "baseline": dict(base_overlay),
            "variant": dict(variant_overlay),
        }
    else:
        normalized = {"baseline": {}, "variant": dict(overlay_payload)}

    baseline_params = dict(baseline)
    baseline_params.update(normalized["baseline"])
    variant_params = dict(baseline_params)
    variant_params.update(normalized["variant"])
    if baseline_params == variant_params:
        raise ValueError("variant overlay does not change any effective parameter")
    return baseline_params, variant_params, normalized


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_result_json(log_lines: list[str]) -> dict[str, Any] | None:
    result = None
    prefix = "RESULT_JSON "
    for line in log_lines:
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        candidate = json.loads(stripped[len(prefix):])
        if not isinstance(candidate, dict):
            raise ValueError("RESULT_JSON payload is not an object")
        result = candidate
    return result


def run_arm(
    arm: str,
    params_path: Path,
    result_path: Path,
    log_path: Path,
    runner: Path,
    python_executable: str,
    extra_runner_args: list[str],
    cwd: Path,
    experiment_sha256: str,
) -> dict[str, Any]:
    command = [
        python_executable,
        str(runner),
        "--fixed",
        "--headless",
        "--params",
        str(params_path),
        *extra_runner_args,
    ]
    print(f"[{arm}] command: {subprocess.list2cmdline(command)}", flush=True)
    # A failed rerun must not leave a prior successful result looking current.
    if result_path.exists():
        result_path.unlink()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    lines = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                print(f"[{arm}] {line}", end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
                lines.append(line)
        return_code = process.wait()
    process_wall_s = time.monotonic() - started
    result = parse_result_json(lines)
    if result is None:
        raise RuntimeError(
            f"{arm} emitted no RESULT_JSON (exit {return_code}); see {log_path}"
        )

    augmented = dict(result)
    augmented.update({
        "ab_arm": arm,
        "ab_process_wall_s": process_wall_s,
        "ab_return_code": return_code,
        "ab_started_at_utc": started_at,
        "ab_params_sha256": _digest(_load_object(params_path, f"{arm} params")),
        "ab_experiment_sha256": experiment_sha256,
    })
    _write_json(result_path, augmented)
    # Keep the standard campaign collector contract. Child output is arm-
    # prefixed above, so this is the only unprefixed harvestable result line.
    print(
        "RESULT_JSON "
        + json.dumps(augmented, ensure_ascii=False, allow_nan=False),
        flush=True,
    )
    if return_code != 0:
        raise RuntimeError(
            f"{arm} emitted a result but exited {return_code}; saved {result_path}"
        )
    return {
        "arm": arm,
        "return_code": return_code,
        "process_wall_s": process_wall_s,
        "result": str(result_path.resolve()),
        "log": str(log_path.resolve()),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="base design/solver params JSON")
    parser.add_argument("--overlay", required=True, help="variant settings overlay JSON")
    parser.add_argument(
        "--arm", choices=("baseline", "variant", "both"), default="both",
        help="run one arm per cluster task, or both sequentially",
    )
    parser.add_argument(
        "--order", choices=("baseline-first", "variant-first"),
        default="baseline-first", help="arm order when --arm=both",
    )
    parser.add_argument("--output-dir", default="efficiency_ab_results")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cwd", default=str(REPO_ROOT))
    parser.add_argument(
        "--runner-arg", action="append", default=[],
        help="extra argument passed to the solver runner (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_params = _load_object(args.params, "params")
    source_overlay = _load_object(args.overlay, "overlay")
    baseline, variant, normalized_overlay = merge_params(
        source_params, source_overlay
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    params_by_arm = {"baseline": baseline, "variant": variant}
    experiment_sha256 = _digest({
        "baseline_params": baseline,
        "variant_params": variant,
    })
    if args.arm == "both":
        selected = (
            ("baseline", "variant")
            if args.order == "baseline-first"
            else ("variant", "baseline")
        )
    else:
        selected = (args.arm,)

    for arm, payload in params_by_arm.items():
        _write_json(output_dir / f"{arm}_params.json", payload)

    manifest: dict[str, Any] = {
        "schema": "mft-efficiency-ab-run-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_params": str(Path(args.params).resolve()),
        "source_overlay": str(Path(args.overlay).resolve()),
        "normalized_overlay": normalized_overlay,
        "baseline_params_sha256": _digest(baseline),
        "variant_params_sha256": _digest(variant),
        "experiment_sha256": experiment_sha256,
        "selected_arms": list(selected),
        "runs": [],
    }
    _write_json(output_dir / "manifest.json", manifest)

    runner = Path(args.runner).resolve()
    cwd = Path(args.cwd).resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"solver runner does not exist: {runner}")
    if not cwd.is_dir():
        raise NotADirectoryError(f"solver working directory does not exist: {cwd}")

    try:
        for arm in selected:
            run_summary = run_arm(
                arm=arm,
                params_path=output_dir / f"{arm}_params.json",
                result_path=output_dir / f"{arm}_result.json",
                log_path=output_dir / f"{arm}.log",
                runner=runner,
                python_executable=args.python,
                extra_runner_args=list(args.runner_arg),
                cwd=cwd,
                experiment_sha256=experiment_sha256,
            )
            manifest["runs"].append(run_summary)
            _write_json(output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        _write_json(output_dir / "manifest.json", manifest)
        raise

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "manifest.json", manifest)
    print(f"A/B outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
