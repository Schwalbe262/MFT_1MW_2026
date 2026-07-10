"""Retired unpinned bulk submitter; all call paths fail closed."""

import requests

SCHEDULER = "http://127.0.0.1:8000"

def submit(name, workdir, run_args, mem_mb=32768, cpus=4):
    """Reject the unpinned project-sync submission path."""
    raise RuntimeError(
        "legacy bulk submission is disabled; use pinned_pilot.py or feeder.py"
    )


def main():
    raise SystemExit(
        "legacy bulk submission is disabled; use pinned_pilot.py or feeder.py"
    )


if __name__ == "__main__":
    main()
