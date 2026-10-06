"""Offline conditional spherical-void PD estimator; never launches AEDT.

Usage: python tools/mft_pd_void_estimate.py --input model_inputs.json --output pd_scenarios.json
The JSON schema and assumptions are documented in module/pd_void_model.py.
All results remain conditional; the command does not qualify PD <= 15 pC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.pd_void_model import estimate_pd_void, estimate_pd_void_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("output must differ from the input evidence file")
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        result = estimate_pd_void_batch(payload) if isinstance(payload, dict) and "models" in payload else estimate_pd_void(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        print(f"Conditional PD estimation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
