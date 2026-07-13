"""Isolate AEDT field-calculator failures for the native-core B**y audit.

This utility reuses an already solved Maxwell project.  It deliberately does
not dispatch Analyze.  Each calculator prefix is rebuilt and registered as a
separate named expression so the first unsupported operation is unambiguous.
On Linux, every AEDT call is bounded by ``--op-timeout-seconds``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = Path(os.environ.get(
    "MFT_PYAEDT_LIBRARY_ROOT", REPO_ROOT.parent / "pyaedt_library"
)).resolve()
LIBRARY_SRC = LIBRARY_ROOT if LIBRARY_ROOT.name == "src" else LIBRARY_ROOT / "src"
if str(LIBRARY_SRC) not in sys.path:
    sys.path.insert(0, str(LIBRARY_SRC))


class AedtOperationTimeout(TimeoutError):
    """Raised when one calculator call exceeds the diagnostic time budget."""


@contextmanager
def _operation_timeout(seconds: float):
    seconds = float(seconds)
    if seconds <= 0 or os.name == "nt":
        yield
        return

    def _expired(_signum, _frame):
        raise AedtOperationTimeout(
            f"AEDT calculator operation exceeded {seconds:g} seconds"
        )

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _cases(object_name: str, reference_name: str, exponent: float):
    """Return complete calculator prefixes in increasing-complexity order."""
    base = [
        ("EnterQty(B)", "EnterQty", ("B",)),
        ("CalcOp(ComplxPeak)", "CalcOp", ("ComplxPeak",)),
    ]
    normalized = [
        *base,
        (
            f"EnterScalarFunc({reference_name})",
            "EnterScalarFunc",
            (reference_name,),
        ),
        ("CalcOp(/)", "CalcOp", ("/",)),
    ]
    powered = [
        *normalized,
        (f"EnterScalar({exponent:g})", "EnterScalar", (float(exponent),)),
        ("CalcOp(Pow)", "CalcOp", ("Pow",)),
    ]
    integrated = [
        *powered,
        (f"EnterVol({object_name})", "EnterVol", (object_name,)),
        ("CalcOp(Integrate)", "CalcOp", ("Integrate",)),
    ]
    return (
        ("diag_bpeak", base),
        ("diag_bnorm_1tesla", normalized),
        ("diag_bpow", powered),
        ("diag_bpow_integral", integrated),
    )


def _recent_messages(desktop, project_name: str, design_name: str):
    try:
        messages = desktop.odesktop.GetMessages(
            project_name, design_name, 0,
        )
    except Exception as error:  # diagnostic best effort only
        return [f"GetMessages failed: {type(error).__name__}: {error}"]
    return [str(item) for item in list(messages or [])[-30:]]


def _run_call(reporter, label: str, method: str, args: tuple, timeout: float):
    started = time.monotonic()
    print(f"BPOW_DIAG_STEP_START {label}", flush=True)
    with _operation_timeout(timeout):
        result = getattr(reporter, method)(*args)
    elapsed = time.monotonic() - started
    print(
        "BPOW_DIAG_STEP_PASS "
        f"{label} elapsed_seconds={elapsed:.6f} result={result!r}",
        flush=True,
    )
    return result, elapsed


def diagnose(args) -> dict:
    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_file() or project_path.suffix.lower() != ".aedt":
        raise FileNotFoundError(f"solved AEDT project is unavailable: {project_path}")
    if not (0 < args.exponent < 10):
        raise ValueError("--exponent must be between 0 and 10")

    from ansys.aedt.core import settings
    from pyaedt_module.core import pyDesktop

    settings.skip_license_check = True
    settings.wait_for_license = False
    desktop = None
    project = None
    evidence = {
        "schema": "mft-1k101-bpow-step-diagnostic-v1",
        "project": str(project_path),
        "design": args.design,
        "object": args.object,
        "reference_variable": args.reference_variable,
        "exponent": float(args.exponent),
        "op_timeout_seconds": float(args.op_timeout_seconds),
        "analyze_dispatched": False,
        "cases": [],
        "passed": False,
    }
    try:
        desktop = pyDesktop(
            version=None,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=True,
        )
        project = desktop.load_project(path=str(project_path))
        project_name = str(project.GetName())
        design = project.SetActiveDesign(args.design)
        if design is None:
            raise RuntimeError(f"failed to activate design {args.design!r}")
        reporter = design.GetModule("FieldsReporter")

        for expression_name, operations in _cases(
            args.object, args.reference_variable, args.exponent
        ):
            case = {
                "expression": expression_name,
                "steps": [],
                "registered": False,
            }
            evidence["cases"].append(case)
            with _operation_timeout(args.op_timeout_seconds):
                reporter.CalcStack("clear")
            try:
                if reporter.DoesNamedExpressionExists(expression_name):
                    reporter.DeleteNamedExpr(expression_name)
            except Exception:
                pass

            for label, method, call_args in operations:
                step = {"label": label, "passed": False}
                case["steps"].append(step)
                try:
                    result, elapsed = _run_call(
                        reporter,
                        label,
                        method,
                        call_args,
                        args.op_timeout_seconds,
                    )
                except Exception as error:
                    step.update({
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    case["failed_step"] = label
                    case["messages"] = _recent_messages(
                        desktop, project_name, args.design
                    )
                    raise
                step.update({
                    "passed": True,
                    "elapsed_seconds": elapsed,
                    "result": repr(result),
                })

            label = f"AddNamedExpression({expression_name})"
            try:
                result, elapsed = _run_call(
                    reporter,
                    label,
                    "AddNamedExpression",
                    (expression_name, "Fields"),
                    args.op_timeout_seconds,
                )
                if result is False:
                    raise RuntimeError("AddNamedExpression returned False")
                with _operation_timeout(args.op_timeout_seconds):
                    exists = bool(reporter.DoesNamedExpressionExists(expression_name))
                if not exists:
                    raise RuntimeError("registered expression is absent on readback")
            except Exception as error:
                case.update({
                    "failed_step": label,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "messages": _recent_messages(
                        desktop, project_name, args.design
                    ),
                })
                raise
            case.update({
                "registered": True,
                "registration_elapsed_seconds": elapsed,
            })

        evidence["passed"] = True
        return evidence
    except Exception as error:
        evidence.update({
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        return evidence
    finally:
        if project is not None:
            try:
                project.Close()
            except Exception:
                pass
        if desktop is not None:
            try:
                desktop.release_desktop(
                    close_projects=True, close_on_exit=True,
                )
            except Exception:
                pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--design", default="maxwell_matrix1")
    parser.add_argument("--object", default="core_2_leg_left")
    parser.add_argument("--reference-variable", default="B_power_reference")
    parser.add_argument("--exponent", type=float, default=1.74)
    parser.add_argument("--op-timeout-seconds", type=float, default=45.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    evidence = diagnose(parse_args(argv))
    print(
        "BPOW_DIAG_JSON "
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0 if evidence.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
