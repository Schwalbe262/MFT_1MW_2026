"""Fail-closed current-seven-temperature Tier-1 corrected-model preflight.

The historical all-11 feedback wrapper is intentionally untouched.  This
module binds the direct corrected generation to the current ``run_nsga2``
module, restores the trained cooling controls hidden by the simple current
problem, and adds every simultaneous physical hard gate omitted there.

This first adapter revision is authentication/smoke only.  It does not install
the old robust offspring projection/repair, so receipts are explicitly not
eligible to launch a full NSGA seed or a Slurm task.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

try:
    from tier1_corrected_generation_adapter import (
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_STAGE_HARD_CONTRACT,
        CURRENT_STAGE_HARD_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_CONTRACT,
        CURRENT_TEMPERATURE_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        adapter_manifest,
        authenticate_code_root,
        authenticate_corrected_generation,
        canonical_sha256,
        process_model_cache,
        sha256_file,
        validate_adapter_manifest,
    )
except ImportError:  # pragma: no cover - repository module import path
    from tools.tier1_corrected_generation_adapter import (
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_STAGE_HARD_CONTRACT,
        CURRENT_STAGE_HARD_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_CONTRACT,
        CURRENT_TEMPERATURE_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        adapter_manifest,
        authenticate_code_root,
        authenticate_corrected_generation,
        canonical_sha256,
        process_model_cache,
        sha256_file,
        validate_adapter_manifest,
    )


RECEIPT_SCHEMA = "mft-tier1-corrected-generation-smoke-receipt-v1"
RUNNER_SCHEMA = "mft-tier1-current7-corrected-runner-v1"
PROBLEM_SCHEMA = "mft-tier1-current7-hard-problem-v1"
BIG = 1e6

BASE_CONSTRAINT_NAMES = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{target}" for target in CURRENT_TEMPERATURE_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "secondary_vertical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
)
ADDITIVE_HARD_CONSTRAINT_NAMES = (
    "minimum_physical_insulation",
    "core_group_manufacturability_limit",
    "half_magnetizing_resonance_minimum",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)
CURRENT7_CONSTRAINT_NAMES = BASE_CONSTRAINT_NAMES + ADDITIVE_HARD_CONSTRAINT_NAMES
SIDE_TEMPERATURE_TARGET = "Tprobe_Rx_side_leeward_max"

CURRENT_STAGE_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "T_limit_C": 110.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "n_core_group_max": 4,
    "primary_conductor_thickness_mm": 5.0,
    "resonance_min_Hz": 15_000.0,
    "magnetizing_inductance_factor": 0.5,
    "size_W_max_mm": 1_200.0,
    "size_L_max_mm": 1_200.0,
    "size_H_max_mm": 750.0,
}
CURRENT_STAGE_SPEC_SHA256 = canonical_sha256(CURRENT_STAGE_SPEC)

EXPECTED_SIMPLE_BASE_FIXED_STACK_MM = {
    "core_plate_t": 20.0,
    "wcp_t": 20.0,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}
VARIABLE_COOLING_DIMENSIONS = (
    "core_plate_t",
    "wcp_t",
    "wcp_len_pct",
)
FIXED_COOLING_PADS_MM = {
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}

PHYSICAL_INSULATION_COLUMNS = (
    "cc_w2c_space_x",
    "cc_w2c_space_y",
    "w2c_w1c_space_x",
    "w2c_w1c_space_y",
    "w1c_w2s_gap_x_actual",
    "w1s_cs_space_x",
    "cs_w1s_space_y",
    "h_gap2",
)
SIDE_PHYSICAL_INSULATION_COLUMNS = (
    "w2s_w1s_space_x",
    "w1s_w2s_space_y",
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def _positive_number(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number <= 0.0:
        raise RuntimeError(f"{label} must be positive")
    return number


def _row_value(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"decoded row is missing {name}") from None


def _frame_row(frame: Any, index: int) -> Any:
    if hasattr(frame, "iloc"):
        return frame.iloc[index]
    return frame[index]


def minimum_physical_insulation_violation(
    frame: Any,
    minimum_mm: float,
    *,
    invalid_value: float = BIG,
) -> Any:
    """Evaluate every realized clearance, including conditional side gaps."""

    import numpy as np

    minimum = _positive_number(minimum_mm, "minimum physical insulation")
    invalid = _positive_number(invalid_value, "invalid constraint value")
    result = np.full(len(frame), invalid, dtype=float)
    for index in range(len(frame)):
        row = _frame_row(frame, index)
        try:
            observed = [
                _finite_number(_row_value(row, name), name)
                for name in PHYSICAL_INSULATION_COLUMNS
            ]
            n1_side = _finite_number(_row_value(row, "N1_side"), "N1_side")
            if n1_side < 0.0:
                raise RuntimeError("N1_side must be non-negative")
            if n1_side > 0.0:
                observed.extend(
                    _finite_number(_row_value(row, name), name)
                    for name in SIDE_PHYSICAL_INSULATION_COLUMNS
                )
            result[index] = max(minimum - value for value in observed)
        except RuntimeError:
            result[index] = invalid
    return result


def apply_current7_side_temperature_condition(
    violation: Any,
    frame: Any,
    *,
    invalid_value: float = BIG,
) -> Any:
    """Apply finite-N2-side activation to the one current side target."""

    import numpy as np

    values = np.asarray(violation, dtype=float).reshape(-1)
    if len(values) != len(frame):
        raise RuntimeError("side-temperature violation/frame length mismatch")
    result = np.full(len(values), float(invalid_value), dtype=float)
    for index, value in enumerate(values):
        try:
            side_turns = _finite_number(
                _row_value(_frame_row(frame, index), "N2_side"),
                "N2_side",
            )
            if side_turns < 0.0:
                raise RuntimeError("N2_side must be non-negative")
            result[index] = value if side_turns > 0.0 else -float(invalid_value)
        except RuntimeError:
            result[index] = float(invalid_value)
    return result


def derive_half_magnetizing_self_resonance(
    measurements: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    magnetizing_inductance_factor: float = 0.5,
) -> dict[str, float]:
    """Derive the authoritative Tx/Rx-only half-Lm resonance screen."""

    if not isinstance(measurements, Mapping) or not callable(
        getattr(params, "get", None)
    ):
        raise RuntimeError("resonance inputs must be mappings")
    leakage_h = _positive_number(
        measurements.get("Llt_phys"), "Llt_phys"
    ) * 1e-6
    coupling = _finite_number(measurements.get("k"), "k")
    if not 0.0 < coupling < 1.0:
        raise RuntimeError("k must satisfy 0 < k < 1")
    c_tx = _positive_number(measurements.get("C_tx_tx_F"), "C_tx_tx_F")
    c_rx = _positive_number(measurements.get("C_rx_rx_F"), "C_rx_rx_F")
    factor = _positive_number(
        magnetizing_inductance_factor,
        "magnetizing inductance factor",
    )
    if factor > 1.0:
        raise RuntimeError("magnetizing inductance factor must be <= 1")
    n1 = _finite_number(params.get("N1_main", 0.0), "N1_main") + _finite_number(
        params.get("N1_side", 0.0), "N1_side"
    )
    n2 = _finite_number(params.get("N2_main", 0.0), "N2_main") + _finite_number(
        params.get("N2_side", 0.0), "N2_side"
    )
    if n1 <= 0.0 or n2 <= 0.0:
        raise RuntimeError("positive primary and secondary turns are required")

    tx_self_h = leakage_h / (1.0 - coupling * coupling)
    tx_magnetizing_h = tx_self_h * coupling * coupling
    rx_magnetizing_h = tx_magnetizing_h * (n2 / n1) ** 2

    def lc_hz(inductance_h: float, capacitance_f: float) -> float:
        frequency = 1.0 / (
            2.0 * math.pi * math.sqrt(inductance_h * capacitance_f)
        )
        return _positive_number(frequency, "self resonance frequency")

    tx_frequency = lc_hz(factor * tx_magnetizing_h, c_tx)
    rx_frequency = lc_hz(factor * rx_magnetizing_h, c_rx)
    return {
        "f_res_tx_half_magnetizing_Hz": tx_frequency,
        "f_res_rx_half_magnetizing_Hz": rx_frequency,
        "f_res_min_tx_rx_only_Hz": min(tx_frequency, rx_frequency),
        "magnetizing_inductance_factor": factor,
        "interwinding_resonance_included": False,
    }


def create_current7_problem_class(
    *,
    base_problem_class: Any,
    base_constraint_names: Any,
    base_fixed_stack_mm: Mapping[str, Any],
    sobol_dims: Any,
    bounding_box_lit: Any,
) -> type:
    """Create an additive wrapper without importing or modifying legacy code."""

    import numpy as np

    if tuple(base_constraint_names) != BASE_CONSTRAINT_NAMES:
        raise RuntimeError("current7 simple-base constraint schema mismatch")
    if dict(base_fixed_stack_mm) != EXPECTED_SIMPLE_BASE_FIXED_STACK_MM:
        raise RuntimeError("current7 simple-base fixed-stack schema mismatch")
    normalized_dims = tuple(
        (str(name), float(lower), float(upper))
        for name, lower, upper in sobol_dims
    )
    names = [item[0] for item in normalized_dims]
    if len(names) != len(set(names)):
        raise RuntimeError("current7 Sobol schema contains duplicate dimensions")
    for name in VARIABLE_COOLING_DIMENSIONS:
        if names.count(name) != 1:
            raise RuntimeError(f"current7 Sobol schema has no unique {name}")
        _, lower, upper = normalized_dims[names.index(name)]
        if not lower < upper:
            raise RuntimeError(f"current7 cooling dimension is not variable: {name}")

    class Current7Tier1Problem(base_problem_class):
        problem_schema = PROBLEM_SCHEMA
        temperature_contract = CURRENT_TEMPERATURE_CONTRACT
        temperature_contract_sha256 = CURRENT_TEMPERATURE_CONTRACT_SHA256
        hard_constraint_contract = CURRENT_STAGE_HARD_CONTRACT
        hard_constraint_contract_sha256 = CURRENT_STAGE_HARD_CONTRACT_SHA256
        stage_spec_sha256 = CURRENT_STAGE_SPEC_SHA256
        offspring_physics_repair = False
        launch_eligible = False

        def __init__(
            self,
            models: Mapping[str, Any],
            spec: Mapping[str, Any] | None = None,
            density_gate: Any = None,
            fixed_overrides: Mapping[str, Any] | None = None,
        ) -> None:
            effective_spec = dict(CURRENT_STAGE_SPEC)
            for name, value in dict(spec or {}).items():
                if name in CURRENT_STAGE_SPEC and not math.isclose(
                    _finite_number(value, name),
                    float(CURRENT_STAGE_SPEC[name]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"Tier-1 hard stage forbids overriding {name}")
                effective_spec[name] = value

            overrides = dict(fixed_overrides or {})
            for name in VARIABLE_COOLING_DIMENSIONS:
                if name in overrides:
                    raise ValueError(
                        f"Tier-1 corrected search requires variable cooling: {name}"
                    )
            required_fixed = {
                **FIXED_COOLING_PADS_MM,
                "cw1": CURRENT_STAGE_SPEC[
                    "primary_conductor_thickness_mm"
                ],
            }
            for name, expected in required_fixed.items():
                if name in overrides and not np.isclose(
                    _finite_number(overrides[name], name),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Tier-1 corrected search fixes {name}={expected:g}"
                    )
            overrides.update(required_fixed)
            super().__init__(
                models,
                spec=effective_spec,
                density_gate=density_gate,
                fixed_overrides=overrides,
            )
            if tuple(self.constraint_names) != BASE_CONSTRAINT_NAMES:
                raise RuntimeError("current7 base constraint schema drifted")
            if int(self.n_ieq_constr) != len(BASE_CONSTRAINT_NAMES):
                raise RuntimeError("current7 base constraint width drifted")
            for name, expected in CURRENT_STAGE_SPEC.items():
                if not math.isclose(
                    _finite_number(self.spec.get(name), name),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(f"Tier-1 stage spec escaped: {name}")

            # The simple current problem overwrites both trained plate controls
            # and clamps their unit coordinates.  Remove only those known exact
            # mutations; pads and cw1 remain fixed and are attested after decode.
            for name in ("core_plate_t", "wcp_t"):
                if not np.isclose(
                    _finite_number(self.fixed_overrides.get(name), name),
                    EXPECTED_SIMPLE_BASE_FIXED_STACK_MM[name],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"unexpected base cooling override: {name}")
                self.fixed_overrides.pop(name)
                coordinate = names.index(name)
                self.xl[coordinate] = 0.0
                self.xu[coordinate] = 1.0
            for name in VARIABLE_COOLING_DIMENSIONS:
                if name in self.fixed_overrides:
                    raise RuntimeError(f"cooling control remained fixed: {name}")
                coordinate = names.index(name)
                if not (
                    np.isclose(self.xl[coordinate], 0.0)
                    and np.isclose(self.xu[coordinate], 1.0)
                ):
                    raise RuntimeError(f"cooling unit bounds remained clamped: {name}")
            for name, expected in required_fixed.items():
                if not np.isclose(
                    _finite_number(self.fixed_overrides.get(name), name),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"fixed manufacturing control escaped: {name}")

            self.base_constraint_names = BASE_CONSTRAINT_NAMES
            self.constraint_names = CURRENT7_CONSTRAINT_NAMES
            self.n_ieq_constr = len(CURRENT7_CONSTRAINT_NAMES)
            self.variable_cooling_dimensions = VARIABLE_COOLING_DIMENSIONS
            self.fixed_cooling_pads_mm = dict(FIXED_COOLING_PADS_MM)
            self._last_decode = None
            self._prediction_cache = None

        def decode_batch(self, values: Any) -> tuple[Any, Any, Any]:
            frame, shrink, valid = super().decode_batch(values)
            valid = np.asarray(valid, dtype=bool).reshape(-1)
            if len(valid) != len(values) or len(frame) != len(values):
                raise RuntimeError("current7 decoder returned an invalid row count")
            for index in np.flatnonzero(valid):
                row = _frame_row(frame, int(index))
                for name, expected in {
                    "cw1": CURRENT_STAGE_SPEC[
                        "primary_conductor_thickness_mm"
                    ],
                    **FIXED_COOLING_PADS_MM,
                }.items():
                    if not np.isclose(
                        _finite_number(_row_value(row, name), name),
                        expected,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise RuntimeError(f"post-decode fixed control escaped: {name}")
                for name in VARIABLE_COOLING_DIMENSIONS:
                    _finite_number(_row_value(row, name), name)
            self._last_decode = (frame, np.asarray(shrink, dtype=float), valid)
            return frame, shrink, valid

        def _predict(self, target: str, frame: Any) -> tuple[Any, Any]:
            cache_key = (str(target), tuple(getattr(frame, "index", range(len(frame)))))
            if self._prediction_cache is not None and cache_key in self._prediction_cache:
                return self._prediction_cache[cache_key]
            mean, half_width = super()._predict(target, frame)
            mean = np.asarray(mean, dtype=float).reshape(-1)
            half_width = np.asarray(half_width, dtype=float).reshape(-1)
            if (
                len(mean) != len(frame)
                or len(half_width) != len(frame)
                or not np.isfinite(mean).all()
                or not np.isfinite(half_width).all()
                or np.any(half_width < 0.0)
            ):
                raise RuntimeError(
                    f"surrogate returned invalid q90 half-width output: {target}"
                )
            result = (mean, half_width)
            if self._prediction_cache is not None:
                self._prediction_cache[cache_key] = result
            return result

        def _evaluate(self, values: Any, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            self._last_decode = None
            self._prediction_cache = {}
            try:
                super()._evaluate(values, out, *args, **kwargs)
                if self._last_decode is None:
                    raise RuntimeError("current7 base evaluation bypassed the decoder")
                frame, _shrink, valid = self._last_decode
                objectives = np.asarray(out.get("F"), dtype=float)
                constraints = np.asarray(out.get("G"), dtype=float)
                expected_shape = (len(values), len(CURRENT7_CONSTRAINT_NAMES))
                if objectives.shape != (len(values), 2):
                    raise RuntimeError("current7 objective shape mismatch")
                if constraints.shape != expected_shape:
                    raise RuntimeError("current7 constraint shape mismatch")
                additive = constraints[:, len(BASE_CONSTRAINT_NAMES):]
                if not np.all(additive == BIG):
                    raise RuntimeError("current7 base wrote into additive hard constraints")

                indices = np.flatnonzero(valid)
                if len(indices):
                    sub = frame.iloc[indices] if hasattr(frame, "iloc") else [
                        frame[int(index)] for index in indices
                    ]
                    side_index = self.constraint_names.index(
                        f"temperature_robust_limit:{SIDE_TEMPERATURE_TARGET}"
                    )
                    constraints[indices, side_index] = (
                        apply_current7_side_temperature_condition(
                            constraints[indices, side_index], sub, invalid_value=BIG
                        )
                    )
                    first_additive = len(BASE_CONSTRAINT_NAMES)
                    constraints[indices, first_additive] = (
                        minimum_physical_insulation_violation(
                            sub,
                            self.spec["insulation_min_mm"],
                            invalid_value=BIG,
                        )
                    )

                    for local_index, global_index in enumerate(indices):
                        row = _frame_row(sub, local_index)
                        try:
                            group_count = _finite_number(
                                _row_value(row, "n_core_group"),
                                "n_core_group",
                            )
                            constraints[
                                global_index, first_additive + 1
                            ] = group_count - float(self.spec["n_core_group_max"])
                        except RuntimeError:
                            constraints[global_index, first_additive + 1] = BIG

                    mean_llt, _ = self._predict("Llt_phys", sub)
                    mean_k, _ = self._predict("k", sub)
                    mean_c_tx, _ = self._predict("C_tx_tx_F", sub)
                    mean_c_rx, _ = self._predict("C_rx_rx_F", sub)
                    for local_index, global_index in enumerate(indices):
                        row = _frame_row(sub, local_index)
                        try:
                            screen = derive_half_magnetizing_self_resonance(
                                {
                                    "Llt_phys": mean_llt[local_index],
                                    "k": mean_k[local_index],
                                    "C_tx_tx_F": mean_c_tx[local_index],
                                    "C_rx_rx_F": mean_c_rx[local_index],
                                },
                                row,
                                magnetizing_inductance_factor=self.spec[
                                    "magnetizing_inductance_factor"
                                ],
                            )
                            minimum = screen["f_res_min_tx_rx_only_Hz"]
                            constraints[
                                global_index, first_additive + 2
                            ] = float(self.spec["resonance_min_Hz"]) - minimum
                        except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
                            constraints[global_index, first_additive + 2] = BIG

                        try:
                            _volume, dimensions = bounding_box_lit(row)
                            dimensions = tuple(
                                _finite_number(value, "exterior dimension")
                                for value in dimensions
                            )
                            if len(dimensions) != 3:
                                raise RuntimeError("exterior dimension count mismatch")
                            limits = (
                                self.spec["size_W_max_mm"],
                                self.spec["size_L_max_mm"],
                                self.spec["size_H_max_mm"],
                            )
                            constraints[
                                global_index,
                                first_additive + 3:first_additive + 6,
                            ] = [
                                observed - float(limit)
                                for observed, limit in zip(dimensions, limits)
                            ]
                        except (RuntimeError, KeyError, TypeError, ValueError, OverflowError):
                            constraints[
                                global_index,
                                first_additive + 3:first_additive + 6,
                            ] = BIG

                objectives[~np.isfinite(objectives)] = BIG
                constraints[~np.isfinite(constraints)] = BIG
                out["F"] = objectives
                out["G"] = constraints
                out["frame"] = frame
                out["decoder_valid"] = valid
            finally:
                self._prediction_cache = None

    Current7Tier1Problem.__name__ = "Current7Tier1Problem"
    Current7Tier1Problem.__qualname__ = "Current7Tier1Problem"
    return Current7Tier1Problem


@dataclass(frozen=True)
class Current7Modules:
    run_nsga2: Any
    nsga2_problem: Any
    predictor: Any
    train_models: Any
    geometry_metrics: Any
    input_parameter: Any
    evidence: dict[str, Any]


def _path_is_below(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [str(path.resolve()), str(root.resolve())]
        ) == str(root.resolve())
    except (OSError, ValueError):
        return False


def _module_path(module: Any, code_root: Path, label: str) -> Path:
    path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if not path.is_file() or not _path_is_below(path, code_root):
        raise RuntimeError(f"current7 module escaped authenticated code root: {label}")
    return path


def load_current7_modules(code_root: Path) -> Current7Modules:
    """Import and attest the exact current optimizer implementation."""

    root = code_root.resolve(strict=True)
    regression_root = root / "regression_260707"
    training_root = regression_root / "training"
    if not regression_root.is_dir() or not training_root.is_dir():
        raise RuntimeError("authenticated code root has no current7 optimizer tree")
    for path in (root, regression_root, training_root):
        token = str(path)
        if token not in sys.path:
            sys.path.insert(0, token)

    modules = {
        "run_nsga2": importlib.import_module("optimization.run_nsga2"),
        "nsga2_problem": importlib.import_module("optimization.nsga2_problem"),
        "predictor": importlib.import_module("predictor"),
        "train_models": importlib.import_module("train_models"),
        "geometry_metrics": importlib.import_module(
            "optimization.geometry_metrics"
        ),
        "input_parameter": importlib.import_module(
            "module.input_parameter_260706"
        ),
    }
    paths = {
        name: _module_path(module, root, name)
        for name, module in modules.items()
    }
    run_nsga2 = modules["run_nsga2"]
    nsga2_problem = modules["nsga2_problem"]
    predictor = modules["predictor"]
    input_parameter = modules["input_parameter"]
    if tuple(run_nsga2.REQUIRED_MODEL_TARGETS) != CURRENT_REQUIRED_MODEL_TARGETS:
        raise RuntimeError("current run_nsga2 required-model contract mismatch")
    if tuple(nsga2_problem.T_TARGETS) != CURRENT_TEMPERATURE_TARGETS:
        raise RuntimeError("current run_nsga2 temperature target contract mismatch")
    if tuple(nsga2_problem.CONSTRAINT_NAMES) != BASE_CONSTRAINT_NAMES:
        raise RuntimeError("current run_nsga2 base constraint contract mismatch")
    if (
        dict(nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM)
        != EXPECTED_SIMPLE_BASE_FIXED_STACK_MM
    ):
        raise RuntimeError("current run_nsga2 simple cooling clamp contract mismatch")
    if not callable(getattr(run_nsga2, "run_one", None)):
        raise RuntimeError("current run_nsga2 has no run_one interface")
    signature = inspect.signature(predictor.EnsemblePredictor.predict_mu_sigma)
    conformal = signature.parameters.get("conformal")
    if conformal is None or conformal.default is not True:
        raise RuntimeError(
            "corrected predictor no longer defaults to q90 conformal output"
        )
    evidence = {
        "module_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "required_model_targets_sha256": CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        "temperature_contract_sha256": CURRENT_TEMPERATURE_CONTRACT_SHA256,
        "base_constraint_names": list(BASE_CONSTRAINT_NAMES),
        "base_constraint_count": len(BASE_CONSTRAINT_NAMES),
        "simple_base_fixed_stack_mm": EXPECTED_SIMPLE_BASE_FIXED_STACK_MM,
        "predict_mu_sigma_conformal_default": True,
        "run_interface": "optimization.run_nsga2.run_one",
    }
    return Current7Modules(
        run_nsga2=run_nsga2,
        nsga2_problem=nsga2_problem,
        predictor=predictor,
        train_models=modules["train_models"],
        geometry_metrics=modules["geometry_metrics"],
        input_parameter=input_parameter,
        evidence=evidence,
    )


@dataclass
class Current7Tier1Runner:
    authenticated: Any
    code_identity: dict[str, Any]
    modules: Current7Modules
    adapter_evidence: dict[str, Any]
    model_cache: Any
    models: Mapping[str, Any]
    inference_binding: dict[str, Any]
    density_gate: Any
    problem: Any

    @property
    def launch_eligible(self) -> bool:
        return False

    def evaluate_coordinates(self, coordinates: Any) -> dict[str, Any]:
        import numpy as np

        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape[1] != int(self.problem.n_var):
            raise RuntimeError("smoke coordinate schema mismatch")
        if not np.isfinite(values).all():
            raise RuntimeError("smoke coordinates are not finite")
        if np.any(values < self.problem.xl) or np.any(values > self.problem.xu):
            raise RuntimeError("smoke coordinates are outside current problem bounds")
        out: dict[str, Any] = {}
        self.problem._evaluate(values, out)
        objectives = np.asarray(out.get("F"), dtype=float)
        constraints = np.asarray(out.get("G"), dtype=float)
        valid = np.asarray(out.get("decoder_valid"), dtype=bool)
        if (
            objectives.shape != (len(values), 2)
            or constraints.shape != (len(values), len(CURRENT7_CONSTRAINT_NAMES))
            or valid.shape != (len(values),)
            or not np.isfinite(objectives).all()
            or not np.isfinite(constraints).all()
        ):
            raise RuntimeError("current7 smoke evaluation did not seal finite F/G")
        return {
            "F": objectives,
            "G": constraints,
            "frame": out["frame"],
            "decoder_valid": valid,
        }

    def run_one(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "full NSGA launch is disabled until old-7c-equivalent offspring "
            "physics projection/repair is installed and attested"
        )


def build_authenticated_runner(
    *,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
    code_root: Path,
    expected_code_revision: str,
    inference_threads: int = 1,
) -> Current7Tier1Runner:
    """Authenticate, load exactly once, and construct the smoke-only runner."""

    if isinstance(inference_threads, bool) or int(inference_threads) != inference_threads:
        raise ValueError("inference_threads must be an integer")
    inference_threads = int(inference_threads)
    if not 1 <= inference_threads <= 4:
        raise ValueError("inference_threads must be from 1 through 4")

    authenticated = authenticate_corrected_generation(
        generation=generation,
        candidate_path=candidate_path,
        quality_path=quality_path,
    )
    code_identity = authenticate_code_root(code_root, expected_code_revision)
    modules = load_current7_modules(Path(code_identity["path"]))
    manifest = adapter_manifest(authenticated, code_identity=code_identity)
    validate_adapter_manifest(manifest)

    cache = process_model_cache(
        authenticated,
        train_models_module=modules.train_models,
        predictor_class=modules.predictor.EnsemblePredictor,
    )
    models = cache.load()
    if tuple(models) != CURRENT_REQUIRED_MODEL_TARGETS or not cache.loaded_once:
        raise RuntimeError("corrected current7 models were not cached exactly once")
    inference_binding = modules.run_nsga2._bound_surrogate_inference(
        models,
        threads=inference_threads,
    )
    if (
        inference_binding.get("target_count") != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or inference_binding.get("threads_per_model") != inference_threads
    ):
        raise RuntimeError("corrected current7 inference binding mismatch")
    density_gate = modules.run_nsga2.build_density_gate(
        str(authenticated.dataset_path),
        list(models["Llt_phys"].features),
    )
    problem_class = create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
    )
    problem = problem_class(models, density_gate=density_gate)
    if (
        tuple(problem.constraint_names) != CURRENT7_CONSTRAINT_NAMES
        or int(problem.n_ieq_constr) != len(CURRENT7_CONSTRAINT_NAMES)
        or problem.offspring_physics_repair is not False
        or problem.launch_eligible is not False
    ):
        raise RuntimeError("corrected current7 problem contract mismatch")
    return Current7Tier1Runner(
        authenticated=authenticated,
        code_identity=code_identity,
        modules=modules,
        adapter_evidence=manifest,
        model_cache=cache,
        models=models,
        inference_binding=inference_binding,
        density_gate=density_gate,
        problem=problem,
    )


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        indent=1,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_name, path)
    finally:
        if os.path.exists(staged_name):
            os.remove(staged_name)


def load_smoke_coordinate(
    path: Path,
    expected_sha256: str,
    *,
    n_var: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    coordinate_path = path.resolve(strict=True)
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError("smoke coordinate expected SHA-256 is invalid")
    actual = sha256_file(coordinate_path)
    if actual != expected:
        raise RuntimeError("smoke coordinate SHA-256 mismatch")
    values = np.asarray(
        np.load(coordinate_path, allow_pickle=False),
        dtype=float,
    )
    if values.shape != (1, int(n_var)) or not np.isfinite(values).all():
        raise RuntimeError("smoke coordinate must contain exactly one finite row")
    return values, {
        "path": str(coordinate_path),
        "sha256": actual,
        "shape": [1, int(n_var)],
        "coordinate_unit_sha256": canonical_sha256(values[0].tolist()),
    }


def _smoke_every_model(
    models: Mapping[str, Any],
    frame: Any,
) -> dict[str, Any]:
    import numpy as np

    if tuple(models) != CURRENT_REQUIRED_MODEL_TARGETS:
        raise RuntimeError("smoke model target order/set mismatch")
    evidence = {}
    for target in CURRENT_REQUIRED_MODEL_TARGETS:
        try:
            mean, half_width = models[target].predict_mu_sigma(
                frame,
                conformal=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                f"smoke model does not accept explicit conformal=True: {target}"
            ) from exc
        mean = np.asarray(mean, dtype=float).reshape(-1)
        half_width = np.asarray(half_width, dtype=float).reshape(-1)
        if (
            mean.shape != (len(frame),)
            or half_width.shape != (len(frame),)
            or not np.isfinite(mean).all()
            or not np.isfinite(half_width).all()
            or np.any(half_width < 0.0)
        ):
            raise RuntimeError(f"smoke model inference failed: {target}")
        evidence[target] = {
            "mean": float(mean[0]),
            "q90_conformal_half_width": float(half_width[0]),
            "finite": True,
        }
    return {
        "target_count": len(evidence),
        "targets": evidence,
        "all_required_targets_exercised": True,
        "explicit_conformal_argument": True,
        "additional_half_width_multiplier": 1.0,
    }


def build_smoke_receipt(
    *,
    runner: Current7Tier1Runner,
    coordinate_evidence: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    model_smoke: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    if runner.launch_eligible or runner.problem.launch_eligible:
        raise RuntimeError("smoke-only receipt unexpectedly became launch eligible")
    objectives = np.asarray(evaluation["F"], dtype=float)
    constraints = np.asarray(evaluation["G"], dtype=float)
    valid = np.asarray(evaluation["decoder_valid"], dtype=bool)
    frame = evaluation["frame"]
    if (
        objectives.shape != (1, 2)
        or constraints.shape != (1, len(CURRENT7_CONSTRAINT_NAMES))
        or valid.tolist() != [True]
        or len(frame) != 1
    ):
        raise RuntimeError("smoke receipt requires one valid decoded model row")
    row = _frame_row(frame, 0)
    decoded_controls = {
        name: _finite_number(_row_value(row, name), name)
        for name in (
            "cw1",
            "core_plate_t",
            "wcp_t",
            "wcp_len_pct",
            "core_plate_pad_t",
            "wcp_pad_t",
            "n_core_group",
            "N1_main",
            "N1_side",
            "N2_main",
            "N2_side",
        )
    }
    if not math.isclose(
        decoded_controls["cw1"],
        CURRENT_STAGE_SPEC["primary_conductor_thickness_mm"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("smoke decoded cw1 is not fixed at 5 mm")
    manifest = validate_adapter_manifest(dict(runner.adapter_evidence))
    if not runner.model_cache.loaded_once:
        raise RuntimeError("smoke receipt cannot attest one model load")
    if (
        model_smoke.get("target_count") != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or model_smoke.get("all_required_targets_exercised") is not True
    ):
        raise RuntimeError("smoke receipt model inference inventory mismatch")

    model_load = {
        "required_targets": list(CURRENT_REQUIRED_MODEL_TARGETS),
        "required_targets_sha256": CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        "loaded_target_count": len(runner.models),
        "cache_load_calls": runner.model_cache.load_calls,
        "full_generation_authentication_passes": (
            runner.model_cache.full_generation_authentication_passes
        ),
        "models_loaded_once_per_process": runner.model_cache.loaded_once,
        "generation_copy_performed": False,
        "inference_binding": runner.inference_binding,
        "model_smoke_completed": True,
        "model_smoke": dict(model_smoke),
    }
    optimizer_repair = {
        "schema_version": "mft-tier1-current7-physics-repair-pending-v1",
        "stages": {
            "initial_population": False,
            "warm_start": False,
            "every_offspring": False,
            "terminal_physical_replay": False,
        },
        "fixed_primary_turns_repair": False,
        "fixed_primary_turns": None,
        "launch_eligible": False,
        "blocking_reason": (
            "old-7c-equivalent hard-physics projection/repair is not installed"
        ),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "authenticated_model_smoke_passed_launch_blocked",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "runner": {
            "schema_version": RUNNER_SCHEMA,
            "problem_schema": PROBLEM_SCHEMA,
            "run_interface": "optimization.run_nsga2.run_one",
            "full_nsga_executed": False,
            "launch_eligible": False,
        },
        "adapter_manifest": manifest,
        "adapter_manifest_sha256": canonical_sha256(manifest),
        "code_modules": runner.modules.evidence,
        "model_load": model_load,
        "model_load_sha256": canonical_sha256(model_load),
        "problem_contract": {
            "stage_spec": CURRENT_STAGE_SPEC,
            "stage_spec_sha256": CURRENT_STAGE_SPEC_SHA256,
            "temperature_contract": CURRENT_TEMPERATURE_CONTRACT,
            "temperature_contract_sha256": CURRENT_TEMPERATURE_CONTRACT_SHA256,
            "hard_constraint_contract": CURRENT_STAGE_HARD_CONTRACT,
            "hard_constraint_contract_sha256": CURRENT_STAGE_HARD_CONTRACT_SHA256,
            "constraint_names": list(CURRENT7_CONSTRAINT_NAMES),
            "constraint_count": len(CURRENT7_CONSTRAINT_NAMES),
            "base_constraint_count": len(BASE_CONSTRAINT_NAMES),
            "additive_hard_constraint_count": len(
                ADDITIVE_HARD_CONSTRAINT_NAMES
            ),
            "base_secondary_vertical_insulation_retained": True,
            "minimum_physical_insulation_is_authoritative_superset": True,
            "variable_cooling_dimensions": list(VARIABLE_COOLING_DIMENSIONS),
            "fixed_cooling_pads_mm": FIXED_COOLING_PADS_MM,
            "simple_base_20mm_plate_clamp_superseded": True,
            "primary_conductor_enforcement": (
                "post_decode_override_smoke_only"
            ),
            "primary_winding_budget_identity_attested": False,
            "side_temperature_condition_applied": True,
            "q90_additional_multiplier": 1.0,
        },
        "optimizer_repair": optimizer_repair,
        "optimizer_repair_sha256": canonical_sha256(optimizer_repair),
        "smoke": {
            "coordinate": dict(coordinate_evidence),
            "decoder_valid": True,
            "decoded_controls": decoded_controls,
            "objectives": {
                "bounding_box_volume_L": float(objectives[0, 0]),
                "predicted_total_loss_W": float(objectives[0, 1]),
            },
            "physical_constraint_G": {
                name: float(constraints[0, index])
                for index, name in enumerate(CURRENT7_CONSTRAINT_NAMES)
            },
            "finite_objectives": True,
            "finite_constraints": True,
            "design_feasibility_required_for_smoke": False,
        },
        "portability": CURRENT_STAGE_HARD_CONTRACT["portability"],
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "canonical_pointer_write_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_smoke_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("corrected-generation smoke receipt must be an object")
    expected_payload_sha = value.get("payload_sha256")
    payload = dict(value)
    payload.pop("payload_sha256", None)
    if expected_payload_sha != canonical_sha256(payload):
        raise RuntimeError("corrected-generation smoke receipt payload SHA mismatch")
    runner = value.get("runner") or {}
    loading = value.get("model_load") or {}
    problem = value.get("problem_contract") or {}
    repair = value.get("optimizer_repair") or {}
    smoke = value.get("smoke") or {}
    if (
        value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("status")
        != "authenticated_model_smoke_passed_launch_blocked"
        or runner.get("schema_version") != RUNNER_SCHEMA
        or runner.get("problem_schema") != PROBLEM_SCHEMA
        or runner.get("full_nsga_executed") is not False
        or runner.get("launch_eligible") is not False
        or loading.get("required_targets") != list(CURRENT_REQUIRED_MODEL_TARGETS)
        or loading.get("required_targets_sha256")
        != CURRENT_REQUIRED_MODEL_TARGETS_SHA256
        or loading.get("loaded_target_count") != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or loading.get("cache_load_calls") != 1
        or loading.get("full_generation_authentication_passes") != 1
        or loading.get("models_loaded_once_per_process") is not True
        or loading.get("generation_copy_performed") is not False
        or loading.get("model_smoke_completed") is not True
        or (loading.get("model_smoke") or {}).get("target_count")
        != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or (loading.get("model_smoke") or {}).get(
            "all_required_targets_exercised"
        ) is not True
        or set(((loading.get("model_smoke") or {}).get("targets") or {}).keys())
        != set(CURRENT_REQUIRED_MODEL_TARGETS)
        or problem.get("stage_spec") != CURRENT_STAGE_SPEC
        or problem.get("stage_spec_sha256") != CURRENT_STAGE_SPEC_SHA256
        or problem.get("temperature_contract") != CURRENT_TEMPERATURE_CONTRACT
        or problem.get("temperature_contract_sha256")
        != CURRENT_TEMPERATURE_CONTRACT_SHA256
        or problem.get("hard_constraint_contract")
        != CURRENT_STAGE_HARD_CONTRACT
        or problem.get("hard_constraint_contract_sha256")
        != CURRENT_STAGE_HARD_CONTRACT_SHA256
        or problem.get("constraint_names") != list(CURRENT7_CONSTRAINT_NAMES)
        or problem.get("constraint_count") != len(CURRENT7_CONSTRAINT_NAMES)
        or problem.get("minimum_physical_insulation_is_authoritative_superset")
        is not True
        or problem.get("simple_base_20mm_plate_clamp_superseded") is not True
        or problem.get("primary_winding_budget_identity_attested") is not False
        or problem.get("q90_additional_multiplier") != 1.0
        or repair.get("stages") != {
            "initial_population": False,
            "warm_start": False,
            "every_offspring": False,
            "terminal_physical_replay": False,
        }
        or repair.get("fixed_primary_turns_repair") is not False
        or repair.get("launch_eligible") is not False
        or smoke.get("decoder_valid") is not True
        or smoke.get("finite_objectives") is not True
        or smoke.get("finite_constraints") is not True
        or set((smoke.get("physical_constraint_G") or {}).keys())
        != set(CURRENT7_CONSTRAINT_NAMES)
        or value.get("portability") != CURRENT_STAGE_HARD_CONTRACT["portability"]
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "slurm_submission_performed",
                "canonical_pointer_write_performed",
                "production_eligible",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("corrected-generation smoke receipt contract mismatch")
    validate_adapter_manifest(value.get("adapter_manifest"))
    if value.get("adapter_manifest_sha256") != canonical_sha256(
        value["adapter_manifest"]
    ):
        raise RuntimeError("corrected-generation adapter manifest SHA mismatch")
    if value.get("model_load_sha256") != canonical_sha256(loading):
        raise RuntimeError("corrected-generation model-load SHA mismatch")
    if value.get("optimizer_repair_sha256") != canonical_sha256(repair):
        raise RuntimeError("corrected-generation optimizer-repair SHA mismatch")
    return value


def run_smoke_preflight(
    *,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
    code_root: Path,
    expected_code_revision: str,
    coordinate_path: Path,
    coordinate_sha256: str,
    output: Path,
    inference_threads: int = 1,
) -> Path:
    output_root = output.resolve()
    if output_root.exists():
        raise RuntimeError("corrected-generation preflight output must not exist")
    if not output_root.parent.is_dir():
        raise RuntimeError("corrected-generation preflight output parent is missing")

    runner = build_authenticated_runner(
        generation=generation,
        candidate_path=candidate_path,
        quality_path=quality_path,
        code_root=code_root,
        expected_code_revision=expected_code_revision,
        inference_threads=inference_threads,
    )
    for protected in (
        runner.authenticated.generation,
        runner.authenticated.registry,
        Path(runner.code_identity["path"]),
    ):
        if _path_is_below(output_root, protected) or _path_is_below(
            protected, output_root
        ):
            raise RuntimeError("preflight output overlaps authenticated input")

    coordinate, coordinate_evidence = load_smoke_coordinate(
        coordinate_path,
        coordinate_sha256,
        n_var=runner.problem.n_var,
    )
    evaluation = runner.evaluate_coordinates(coordinate)
    if not bool(evaluation["decoder_valid"][0]):
        raise RuntimeError("authenticated smoke coordinate did not decode")
    frame = evaluation["frame"].iloc[[0]]
    model_smoke = _smoke_every_model(runner.models, frame)
    receipt = build_smoke_receipt(
        runner=runner,
        coordinate_evidence=coordinate_evidence,
        evaluation=evaluation,
        model_smoke=model_smoke,
    )
    validate_smoke_receipt(receipt)

    output_root.mkdir()
    receipt_path = output_root / "authentication_receipt.json"
    _atomic_json(receipt_path, receipt)
    validate_smoke_receipt(json.loads(receipt_path.read_text(encoding="utf-8")))
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate and smoke one corrected current7 generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--generation", type=Path, required=True)
    smoke.add_argument("--candidate", type=Path, required=True)
    smoke.add_argument("--quality-status", type=Path, required=True)
    smoke.add_argument("--code-root", type=Path, required=True)
    smoke.add_argument("--expected-code-revision", required=True)
    smoke.add_argument("--coordinate", type=Path, required=True)
    smoke.add_argument("--coordinate-sha256", required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--inference-threads", type=int, default=1)
    validate = subparsers.add_parser("validate-receipt")
    validate.add_argument("receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-receipt":
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_smoke_receipt(receipt)
        print(json.dumps({
            "valid": True,
            "schema_version": receipt["schema_version"],
            "payload_sha256": receipt["payload_sha256"],
            "launch_eligible": False,
        }, sort_keys=True))
        return 0
    receipt_path = run_smoke_preflight(
        generation=args.generation,
        candidate_path=args.candidate,
        quality_path=args.quality_status,
        code_root=args.code_root,
        expected_code_revision=args.expected_code_revision,
        coordinate_path=args.coordinate,
        coordinate_sha256=args.coordinate_sha256,
        output=args.output,
        inference_threads=args.inference_threads,
    )
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
