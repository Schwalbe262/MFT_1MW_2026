"""Build or seal the fixed-N1=6 orthogonal Tier-1 warm handoff.

This command writes only below ``--output``.  It does not contact Slurm,
submit FEA, start AEDT, publish a pointer, or promote a design/model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_fixed_n1_6_bridge_warm_handoff as handoff


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--normalized-root", type=Path, default=handoff.DEFAULT_NORMALIZED,
    )
    result.add_argument(
        "--thermal-root", type=Path, default=handoff.DEFAULT_THERMAL,
    )
    result.add_argument(
        "--bridge-root", type=Path, default=handoff.DEFAULT_BRIDGE,
    )
    result.add_argument(
        "--fixed6-root", type=Path, default=handoff.DEFAULT_FIXED6,
    )
    result.add_argument(
        "--nsga-code-root", type=Path, default=handoff.DEFAULT_CODE,
    )
    result.add_argument(
        "--output", type=Path, default=handoff.DEFAULT_ORTHOGONAL_OUTPUT,
    )
    result.add_argument("--seal-post-repair-preflight-result", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.seal_post_repair_preflight_result is not None:
        value = handoff.seal_orthogonal_post_repair_preflight(
            args.output / handoff.ORTHOGONAL_CONTRACT_FILENAME,
            args.seal_post_repair_preflight_result,
            args.nsga_code_root,
        )
    else:
        value = handoff.build_orthogonal_handoff(
            normalized_root=args.normalized_root,
            thermal_root=args.thermal_root,
            bridge_root=args.bridge_root,
            fixed6_root=args.fixed6_root,
            nsga_code_root=args.nsga_code_root,
            output=args.output,
        )
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
