"""
NSGA-2 멀티 재시작 드라이버.

- pop 200, ftol 수렴 종료 (고정 세대수 대신), 재시작 N회 (시드 다양화)
- warm start: 이전 라운드 Pareto(archive/pareto_X.npy)를 초기 개체군에 주입
- 전역 Pareto 병합 + 전체 평가 아카이브 (AL 인필 선정용)

사용:
  python run_nsga2.py --restarts 16 --round 1
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "training"))

from optimization.nsga2_problem import (  # noqa: E402
    CONSTRAINT_NAMES,
    DEFAULT_SPEC,
    MFTProblem,
    NSGA_FIXED_THERMAL_STACK_MM,
    T_TARGETS,
)
from optimization.design_summary import (  # noqa: E402
    COMPONENT_LOSS_TARGETS,
    pareto_design_summary,
)
from optimization.resonance import derive_resonances  # noqa: E402
from module.input_parameter_260706 import _SOBOL_DIMS  # noqa: E402
from model_targets import SURROGATE_CAPACITANCE_TARGETS  # noqa: E402

REQUIRED_MODEL_TARGETS = [
    "Llt_phys", "k",
    *SURROGATE_CAPACITANCE_TARGETS,
    "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
    *COMPONENT_LOSS_TARGETS,
    # B_max remains a raw FEA diagnostic.  The optimizer uses the analytical
    # bulk volt-second constraint and keeps B_mean only as a comparison model.
    "B_mean_core", *T_TARGETS,
]
MIN_STRICT_FULL_ROWS = 3000
EXPERIMENTAL_MIN_STRICT_FULL_ROWS = 2000
VETTED_QUALITY_THRESHOLDS = os.path.abspath(os.path.join(
    HERE, "..", "training", "model_quality_thresholds.json"
))
DETERMINISTIC_INFEASIBLE_EXIT_CODE = 42


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    staged = f"{path}.{os.getpid()}.tmp"
    try:
        with open(staged, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _population_infeasibility(population, constraint_names=CONSTRAINT_NAMES):
    """Return compact final-population evidence even when pymoo has no result.X."""

    if population is None or not callable(getattr(population, "get", None)):
        return {"available": False, "population_size": 0}
    constraints = np.asarray(population.get("G"), dtype=float)
    chromosomes = np.asarray(population.get("X"), dtype=float)
    if constraints.ndim != 2 or constraints.shape[1] != len(constraint_names):
        raise RuntimeError(
            "NSGA final population constraint width does not match the sealed schema"
        )
    if chromosomes.ndim != 2 or chromosomes.shape[0] != constraints.shape[0]:
        raise RuntimeError("NSGA final population chromosome evidence is invalid")
    positive = np.maximum(constraints, 0.0)
    finite_rows = np.isfinite(constraints).all(axis=1)
    total_violation = np.where(
        finite_rows, positive.sum(axis=1), np.inf
    )
    feasible = finite_rows & (constraints <= 0.0).all(axis=1)
    closest_index = (
        int(np.argmin(total_violation))
        if len(total_violation) and np.isfinite(total_violation).any()
        else None
    )
    per_constraint = []
    for index, name in enumerate(constraint_names):
        values = constraints[:, index]
        finite = values[np.isfinite(values)]
        per_constraint.append({
            "index": index,
            "name": name,
            "finite_count": int(len(finite)),
            "passing_count": int(np.count_nonzero(finite <= 0.0)),
            "minimum_value": _finite_float(np.min(finite)) if len(finite) else None,
            "median_value": _finite_float(np.median(finite)) if len(finite) else None,
            "minimum_violation": (
                _finite_float(np.min(np.maximum(finite, 0.0)))
                if len(finite) else None
            ),
        })
    closest = None
    if closest_index is not None:
        closest = {
            "population_index": closest_index,
            "total_positive_violation": _finite_float(
                total_violation[closest_index]
            ),
            "chromosome_unit": [
                _finite_float(value) for value in chromosomes[closest_index]
            ],
            "constraint_values": {
                name: _finite_float(constraints[closest_index, index])
                for index, name in enumerate(constraint_names)
            },
        }
    return {
        "available": True,
        "population_size": int(constraints.shape[0]),
        "finite_row_count": int(np.count_nonzero(finite_rows)),
        "feasible_count": int(np.count_nonzero(feasible)),
        "closest_candidate": closest,
        "constraints": per_constraint,
    }


def _sobol_schema():
    payload = [
        {"name": name, "lower": float(lower), "upper": float(upper)}
        for name, lower, upper in _SOBOL_DIMS
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return payload, hashlib.sha256(encoded).hexdigest()


def _load_compatible_warm_start(path, n_var):
    if not os.path.isfile(path):
        return None
    warm = np.load(path)
    if warm.ndim != 2 or warm.shape[1] != int(n_var):
        print(
            f"warm start ignored: {path} has shape {warm.shape}, "
            f"expected (*, {n_var})"
        )
        return None
    return warm


def _recomputed_strict_full_rows(
        dataset_path, solver_revision=None, library_revision=None):
    from quality_contract import annotate_validity

    audited = annotate_validity(
        pd.read_parquet(dataset_path),
        expected_solver_revision=solver_revision,
        expected_library_revision=library_revision,
    )
    return int(audited["_strict_valid_full"].sum())


def _experimental_quality_contract(
        quality, quality_path, generation_report, generation_path,
        dataset_path):
    """Authenticate an explicitly non-production 2k active-learning input."""

    if not isinstance(quality, dict):
        raise SystemExit("experimental quality evidence is not an object")
    expected = {
        "schema_version": 1,
        "lane": "provisional_2000_surrogate",
        "passed": False,
        "activation_performed": False,
        "nsga2_enqueued": False,
        "verification_enqueued": False,
        "production_minimum_strict_full_rows": MIN_STRICT_FULL_ROWS,
        "provisional_minimum_strict_full_rows": EXPERIMENTAL_MIN_STRICT_FULL_ROWS,
        "terminal_reason": "provisional_quality_gate_failed",
    }
    mismatches = [
        key for key, value in expected.items() if quality.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            "experimental quality evidence contract mismatch: "
            + ",".join(mismatches)
        )
    blockers = quality.get("failed_targets")
    if not isinstance(blockers, dict) or not blockers:
        raise SystemExit("experimental lane has no sealed quality blockers")
    for target, reasons in blockers.items():
        if not isinstance(target, str) or not target or not isinstance(reasons, list) \
                or not reasons or not all(isinstance(item, str) and item for item in reasons):
            raise SystemExit("experimental quality blocker inventory is invalid")

    dataset_sha = _sha256(dataset_path)
    report_path = os.path.join(generation_path, "train_report.json")
    report_sha = _sha256(report_path)
    if (
            quality.get("dataset_sha256") != dataset_sha
            or generation_report.get("dataset_sha256") != dataset_sha
            or quality.get("generation_report_sha256") != report_sha):
        raise SystemExit(
            "experimental surrogate generation/quality/dataset identity mismatch"
        )
    try:
        quality_rows = int(quality["strict_full_rows"])
        report_rows = int(generation_report["strict_full_rows"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit("experimental strict-full row evidence is invalid")
    if (
            quality_rows < EXPERIMENTAL_MIN_STRICT_FULL_ROWS
            or quality_rows != report_rows):
        raise SystemExit("experimental strict-full row identity mismatch")

    revisions = {}
    for key in ("solver_revision_pin", "library_revision_pin"):
        value = str(quality.get(key) or "").lower()
        if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
            raise SystemExit(f"experimental quality evidence has no pinned {key}")
        revisions[key] = value
    normalized = dict(quality)
    normalized.update({
        "training_run_id": generation_report.get("training_run_id"),
        "dataset_sha256": dataset_sha,
        "solver_revision": revisions["solver_revision_pin"],
        "library_revision": revisions["library_revision_pin"],
        "quality_status_sha256": _sha256(quality_path),
        "generation_report_sha256": report_sha,
    })
    return normalized


def load_models(registry=None, generation=None, *, allow_unaccepted=False):
    from predictor import EnsemblePredictor
    import predictor as predictor_mod
    reg = registry or predictor_mod.REGISTRY
    generation_record = None
    if generation and allow_unaccepted:
        # The experimental lane is explicitly tied to failed quality evidence,
        # so it cannot use EnsemblePredictor.load_generation(), whose contract
        # correctly requires a passing production gate.  Still authenticate
        # every model/report byte through the immutable registry inventory;
        # the experimental quality contract above separately binds that report
        # to the exact dataset, revision pins, row count, and blockers.
        from train_models import load_generation

        generation_record = load_generation(
            reg, generation, require_accepted=False
        )
    models = {}
    for t in REQUIRED_MODEL_TARGETS:
        try:
            if generation_record is not None:
                models[t] = EnsemblePredictor._load_record(
                    t, generation_record
                )
            elif generation:
                models[t] = EnsemblePredictor.load_generation(
                    t, registry=reg, generation=generation
                )
            else:
                models[t] = EnsemblePredictor.load(t, registry=reg)
        except (FileNotFoundError, OSError):
            pass
    return models


def _bound_surrogate_inference(models, threads=1):
    """Disable nested all-core prediction below restart-level parallelism."""

    evidence = []
    for target in sorted(models):
        configure = getattr(models[target], "configure_inference_threads", None)
        if not callable(configure):
            raise RuntimeError(
                f"surrogate predictor cannot bind inference threads: {target}"
            )
        result = configure(threads)
        if not isinstance(result, dict) or result.get("threads") != int(threads):
            raise RuntimeError(
                f"surrogate inference thread attestation failed: {target}"
            )
        evidence.append(result)
    families = sorted({
        family
        for result in evidence
        for family in result.get("families", [])
    })
    return {
        "threads_per_model": int(threads),
        "target_count": len(evidence),
        "model_count": sum(
            int(result.get("model_count") or 0) for result in evidence
        ),
        "families": families,
        "policy": "outer_restart_parallelism_inner_model_serial_v1",
    }


def build_density_gate(dataset_path, features):
    from predictor import DensityGate
    from checkpoint_train import to_physical
    from quality_contract import annotate_validity

    audited = annotate_validity(pd.read_parquet(dataset_path))
    df = to_physical(audited.loc[audited["_strict_valid_full"]])
    if df.empty:
        raise RuntimeError("density gate has no strict-full training rows")
    return DensityGate(df, features)


def nds_merge(F_list, X_list, meta_list):
    """비지배 정렬로 풀 병합"""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    F = np.vstack(F_list)
    X = np.vstack(X_list)
    fronts = NonDominatedSorting().do(F, only_non_dominated_front=True)
    metas = [m for ms in meta_list for m in ms]
    return X[fronts], F[fronts], [metas[i] for i in fronts]


def run_one(problem, seed, pop, warm_X=None, max_gen=600):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    rng = np.random.default_rng(seed)
    if warm_X is not None and len(warm_X):
        n_warm = min(len(warm_X), pop // 2)
        pick = rng.choice(len(warm_X), n_warm, replace=False)
        init = np.vstack([warm_X[pick],
                          rng.random((pop - n_warm, problem.n_var))])
    else:
        init = FloatRandomSampling().do(problem, pop).get("X")

    # Array-based warm/random sampling bypasses pymoo's sampler repair path.
    # Project it explicitly so fixed normalized plate coordinates are honored
    # in the chromosome as well as by the fail-closed physical decoder.
    init = np.minimum(
        np.maximum(np.asarray(init, dtype=float), problem.xl), problem.xu
    )

    algo = NSGA2(pop_size=pop, sampling=init)
    term = DefaultMultiObjectiveTermination(ftol=0.0025, period=30, n_max_gen=max_gen)
    res = minimize(problem, algo, term, seed=int(seed), verbose=False, save_history=False)
    return res


def _run_one_arrays(problem, seed, pop, warm_X, max_gen):
    """Process-pool boundary that returns only compact, pickle-safe evidence."""
    result = run_one(
        problem, seed=seed, pop=pop, warm_X=warm_X, max_gen=max_gen
    )
    final_population = getattr(result, "pop", None)
    if final_population is None:
        final_population = getattr(getattr(result, "algorithm", None), "pop", None)
    return (
        result.X,
        result.F,
        int(result.algorithm.n_gen),
        _population_infeasibility(
            final_population,
            constraint_names=getattr(problem, "constraint_names", CONSTRAINT_NAMES),
        ),
    )


def run_restarts(
    problem, restarts, pop, warm_X=None, workers=4, max_gen=600,
    executor_factory=ProcessPoolExecutor,
):
    """Run deterministic restart seeds concurrently and return seed order.

    Four processes is the production ceiling.  Returning results in restart
    order keeps the merged Pareto and manifest deterministic even though
    futures finish out of order.
    """
    restarts = int(restarts)
    workers = int(workers)
    if restarts < 1:
        raise ValueError("NSGA restarts must be positive")
    if not 1 <= workers <= 4:
        raise ValueError("NSGA workers must be between 1 and 4")
    workers = min(workers, restarts)
    if workers == 1:
        return [
            _run_one_arrays(problem, 1000 + index, pop, warm_X, max_gen)
            for index in range(restarts)
        ]
    ordered = {}
    with executor_factory(max_workers=workers) as executor:
        pending = {
            executor.submit(
                _run_one_arrays,
                problem,
                1000 + index,
                pop,
                warm_X,
                max_gen,
            ): index
            for index in range(restarts)
        }
        for future in as_completed(pending):
            index = pending[future]
            ordered[index] = future.result()
    return [ordered[index] for index in range(restarts)]


def _completed_output_is_valid(out_dir):
    marker_path = os.path.join(out_dir, "COMPLETED")
    manifest_path = os.path.join(out_dir, "optimization_manifest.json")
    if not os.path.isfile(marker_path):
        return False
    try:
        marker = json.load(open(marker_path, encoding="utf-8"))
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        return (
            marker.get("optimization_manifest_sha256") == _sha256(manifest_path)
            and manifest.get("pareto_front_sha256")
            == _sha256(os.path.join(out_dir, "pareto_front.csv"))
            and manifest.get("pareto_X_sha256")
            == _sha256(os.path.join(out_dir, "pareto_X.npy"))
            and manifest.get("pareto_F_sha256")
            == _sha256(os.path.join(out_dir, "pareto_F.npy"))
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument(
        "--workers", type=int, default=4,
        help="parallel restart processes (production maximum: 4)",
    )
    ap.add_argument("--round", type=int, default=0, help="AL 라운드 번호 (출력 디렉토리)")
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data", "dataset", "train.parquet"))
    ap.add_argument("--registry", default=os.path.join(HERE, "..", "training", "registry"))
    ap.add_argument("--registry-generation", default=None)
    ap.add_argument("--quality-status", default=None)
    ap.add_argument(
        "--experimental-quality-status", default=None,
        help=(
            "explicit failed provisional-2000 quality evidence; runs only a "
            "non-production active-learning search"
        ),
    )
    ap.add_argument(
        "--fea-solver-revision",
        default=None,
        help="exact solver commit reserved for downstream FEA verification",
    )
    ap.add_argument(
        "--fea-library-revision",
        default=None,
        help="exact PyAEDT library commit reserved for downstream FEA verification",
    )
    ap.add_argument("--output-root", default=os.path.join(HERE, "..", "al_rounds"))
    ap.add_argument(
        "--quality-thresholds",
        default=VETTED_QUALITY_THRESHOLDS,
    )
    ap.add_argument("--spec", default=None, help="스펙 오버라이드 JSON")
    ap.add_argument("--no-density-gate", action="store_true")
    args = ap.parse_args()

    if args.restarts < 1 or args.pop < 2:
        ap.error("restarts must be positive and population must be at least 2")
    if not 1 <= args.workers <= 4:
        ap.error("workers must be between 1 and 4")

    out_dir = os.path.join(args.output_root, f"round_{args.round:02d}")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "COMPLETED")):
        if _completed_output_is_valid(out_dir):
            print(f"optimization generation already complete: {out_dir}")
            return
        raise SystemExit("completed optimization output failed authentication")

    vetted_thresholds_sha256 = _sha256(VETTED_QUALITY_THRESHOLDS)
    if _sha256(args.quality_thresholds) != vetted_thresholds_sha256:
        raise SystemExit("quality thresholds differ from the vetted production contract")

    experimental = bool(args.experimental_quality_status)
    if experimental and args.quality_status:
        raise SystemExit(
            "production and experimental quality evidence are mutually exclusive"
        )
    supplied_quality = args.experimental_quality_status or args.quality_status
    if bool(args.registry_generation) != bool(supplied_quality):
        raise SystemExit(
            "--registry-generation and one quality-status mode must be supplied together"
        )
    registry_for_models = args.registry
    generation_report = {}
    pinned_generation = None
    if args.registry_generation:
        pinned_generation = os.path.abspath(args.registry_generation)
        with open(supplied_quality, encoding="utf-8") as handle:
            quality = json.load(handle)
        with open(
            os.path.join(pinned_generation, "train_report.json"), encoding="utf-8"
        ) as handle:
            generation_report = json.load(handle)
        dataset_sha = _sha256(args.dataset)
        if experimental:
            quality = _experimental_quality_contract(
                quality,
                os.path.abspath(args.experimental_quality_status),
                generation_report,
                pinned_generation,
                os.path.abspath(args.dataset),
            )
        elif (not quality.get("passed")
                or quality.get("dataset_sha256") != dataset_sha
                or int(quality.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS
                or quality.get("quality_thresholds_sha256")
                != vetted_thresholds_sha256
                or generation_report.get("dataset_sha256") != dataset_sha
                or int(generation_report.get("strict_full_rows") or 0)
                < MIN_STRICT_FULL_ROWS
                or generation_report.get("training_run_id") != quality.get("training_run_id")):
            raise SystemExit("pinned surrogate generation/quality/dataset identity mismatch")
        for key in ("solver_revision", "library_revision"):
            value = str(quality.get(key) or "")
            if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
                raise SystemExit(f"quality snapshot has no pinned {key}")
        verified_strict_rows = _recomputed_strict_full_rows(
            args.dataset,
            quality["solver_revision"],
            quality["library_revision"],
        )
        minimum_rows = (
            EXPERIMENTAL_MIN_STRICT_FULL_ROWS if experimental
            else MIN_STRICT_FULL_ROWS
        )
        if (verified_strict_rows < minimum_rows
                or verified_strict_rows != int(quality["strict_full_rows"])
                or verified_strict_rows
                != int(generation_report["strict_full_rows"])):
            raise SystemExit(
                "pinned dataset strict-full cohort does not match quality metadata"
            )
    else:
        from model_quality_gate import evaluate_registry
        with open(args.quality_thresholds, encoding="utf-8") as handle:
            quality_thresholds = json.load(handle)
        quality = evaluate_registry(args.registry, args.dataset, quality_thresholds)
        quality["quality_thresholds_sha256"] = vetted_thresholds_sha256
        if (not quality["passed"]
                or int(quality.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS):
            raise SystemExit(
                "surrogate quality gate failed: " + "; ".join(quality["reasons"][:20])
            )
        verified_strict_rows = _recomputed_strict_full_rows(args.dataset)
        if (verified_strict_rows < MIN_STRICT_FULL_ROWS
                or verified_strict_rows != int(quality["strict_full_rows"])):
            raise SystemExit(
                "dataset strict-full cohort does not match quality metadata"
            )

    training_solver_revision = str(quality.get("solver_revision") or "").lower()
    training_library_revision = str(quality.get("library_revision") or "").lower()
    fea_solver_revision = str(
        args.fea_solver_revision or training_solver_revision
    ).lower()
    fea_library_revision = str(
        args.fea_library_revision or training_library_revision
    ).lower()
    for label, revision in (
        ("training solver", training_solver_revision),
        ("training library", training_library_revision),
        ("FEA solver", fea_solver_revision),
        ("FEA library", fea_library_revision),
    ):
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise SystemExit(f"optimization has no exact {label} revision")
    if experimental and (
        not args.fea_solver_revision or not args.fea_library_revision
    ):
        raise SystemExit(
            "experimental optimization requires explicit downstream FEA revisions"
        )

    models = load_models(
        registry_for_models,
        generation=pinned_generation,
        allow_unaccepted=experimental,
    )
    missing = [target for target in REQUIRED_MODEL_TARGETS if target not in models]
    if missing:
        raise SystemExit(f"required model generation is incomplete: {missing}")
    inference_parallelism = _bound_surrogate_inference(models, threads=1)
    print(json.dumps({
        "event": "surrogate_inference_parallelism",
        **inference_parallelism,
    }, sort_keys=True), flush=True)
    if pinned_generation:
        generation_artifact_root = pinned_generation
    else:
        from train_models import load_active_generation

        generation_artifact_root = load_active_generation(args.registry)["generation"]

    spec = dict(DEFAULT_SPEC)
    if args.spec:
        spec.update(json.load(open(args.spec)))

    gate = None
    if not args.no_density_gate:
        gate = build_density_gate(args.dataset, models["Llt_phys"].features)

    problem = MFTProblem(models, spec=spec, density_gate=gate)

    # warm start: 이전 라운드 Pareto
    warm = None
    prev = os.path.join(args.output_root, f"round_{args.round - 1:02d}", "pareto_X.npy")
    if os.path.isfile(prev):
        warm = _load_compatible_warm_start(prev, problem.n_var)
    if warm is not None:
        print(f"warm start: {len(warm)} points from round {args.round - 1}")

    F_list, X_list, feasible_counts = [], [], []
    restart_results = run_restarts(
        problem,
        args.restarts,
        args.pop,
        warm_X=warm,
        workers=args.workers,
    )
    restart_diagnostics = []
    for r, (result_x, result_f, generations, diagnostic) in enumerate(restart_results):
        restart_diagnostics.append({
            "restart": r,
            "seed": 1000 + r,
            "generations": generations,
            **diagnostic,
        })
        if result_x is None:
            feasible_counts.append(0)
            continue
        X = np.atleast_2d(result_x)
        F = np.atleast_2d(result_f)
        X_list.append(X)
        F_list.append(F)
        feasible_counts.append(len(X))
        print(f"restart {r}: {len(X)} pareto pts, vol {F[:,0].min():.0f}-{F[:,0].max():.0f}L, "
              f"loss {F[:,1].min():.0f}-{F[:,1].max():.0f}W, gen {generations}")

    if not X_list:
        report_path = os.path.join(out_dir, "infeasibility_report.json")
        report = {
            "schema_version": "mft-nsga2-infeasibility-v1",
            "terminal_reason": "deterministic_no_feasible_solution",
            "exit_code": DETERMINISTIC_INFEASIBLE_EXIT_CODE,
            "production_eligible": False,
            "experimental_active_learning": experimental,
            "dataset_path": os.path.abspath(args.dataset),
            "dataset_sha256": _sha256(args.dataset),
            "strict_full_rows": verified_strict_rows,
            "registry_generation": os.path.abspath(generation_artifact_root),
            "generation_report_sha256": _sha256(os.path.join(
                generation_artifact_root, "train_report.json"
            )),
            "quality_status_sha256": (
                _sha256(supplied_quality) if supplied_quality else None
            ),
            "training_solver_revision": training_solver_revision,
            "training_library_revision": training_library_revision,
            "fea_solver_revision": fea_solver_revision,
            "fea_library_revision": fea_library_revision,
            "optimization_source_sha256": _sha256(__file__),
            "problem_source_sha256": _sha256(os.path.join(
                HERE, "nsga2_problem.py"
            )),
            "constraint_names": list(CONSTRAINT_NAMES),
            "effective_spec": spec,
            "fixed_thermal_stack_mm": NSGA_FIXED_THERMAL_STACK_MM,
            "density_gate_enabled": gate is not None,
            "restarts": args.restarts,
            "population": args.pop,
            "workers": min(args.workers, args.restarts),
            "seeds": [1000 + index for index in range(args.restarts)],
            "termination": {
                "ftol": 0.0025, "period": 30, "max_generations": 600,
            },
            "restart_diagnostics": restart_diagnostics,
        }
        _atomic_json(report_path, report)
        print(json.dumps({
            "event": "deterministic_no_feasible_solution",
            "exit_code": DETERMINISTIC_INFEASIBLE_EXIT_CODE,
            "infeasibility_report": os.path.abspath(report_path),
            "infeasibility_report_sha256": _sha256(report_path),
        }, sort_keys=True), flush=True)
        raise SystemExit(DETERMINISTIC_INFEASIBLE_EXIT_CODE)

    Xp, Fp, _ = nds_merge(F_list, X_list, [[None] * len(x) for x in X_list])
    pareto_x_path = os.path.join(out_dir, "pareto_X.npy")
    pareto_f_path = os.path.join(out_dir, "pareto_F.npy")
    np.save(pareto_x_path, Xp)
    np.save(pareto_f_path, Fp)

    # Pareto 후보의 파라미터/예측치 테이블
    frame, shrink, valid = problem.decode_batch(Xp)
    rows = []
    for i in range(len(Xp)):
        row = {"volume_L": Fp[i, 0], "total_loss_W": Fp[i, 1]}
        predictions = {}
        for t in REQUIRED_MODEL_TARGETS:
            mdl = models[t]
            mu, sg = mdl.predict_mu_sigma(frame.iloc[[i]])
            predictions[t] = float(mu[0])
            row[f"pred_{t}"] = predictions[t]
            row[f"sigma_{t}"] = float(sg[0])
        for target, value in derive_resonances(
            predictions, frame.iloc[i].to_dict()
        ).items():
            row[f"pred_{target}"] = value
        summary = pareto_design_summary(
            frame.iloc[i],
            predictions,
            Fp[i, 1],
            leakage_target_uH=spec["Llt_target_uH"],
            core_lamination_factor=spec["core_lamination_factor"],
            B_area_basis=spec["B_area_basis"],
        )
        if not np.isclose(
            summary["volume_L"], Fp[i, 0], rtol=1e-9, atol=1e-9
        ):
            raise RuntimeError(
                "Pareto volume objective does not match decoded geometry"
            )
        # Some requested audit names (Ae_m2, n_core_group, wcp_len_pct)
        # already exist in the authoritative decoded frame. Cross-check them
        # and retain that original column once instead of emitting ambiguous
        # duplicate CSV headers.
        overlap = set(summary).intersection(frame.columns)
        for column in overlap:
            decoded_value = float(frame.iloc[i][column])
            summary_value = float(summary[column])
            if not np.isclose(
                decoded_value, summary_value, rtol=1e-12, atol=1e-12
            ):
                raise RuntimeError(
                    f"Pareto summary {column} does not match decoded geometry"
                )
        row.update({
            key: value for key, value in summary.items()
            if key not in overlap
        })
        rows.append(row)
    front_path = os.path.join(out_dir, "pareto_front.csv")
    pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1) \
        .to_csv(front_path, index=False)

    model_artifacts = {
        target: {
            name: _sha256(os.path.join(generation_artifact_root, target, name))
            for name in ("models.pkl", "meta.json")
        }
        for target in REQUIRED_MODEL_TARGETS
    }

    sobol_schema, sobol_schema_sha256 = _sobol_schema()
    manifest_path = os.path.join(out_dir, "optimization_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({
            "training_run_id": quality.get("training_run_id"),
            "dataset_sha256": quality.get("dataset_sha256"),
            "quality_gate_passed": not experimental,
            "experimental_active_learning": experimental,
            "production_eligible": not experimental,
            "quality_blockers": (
                quality.get("failed_targets") if experimental else {}
            ),
            "experimental_minimum_strict_full_rows": (
                EXPERIMENTAL_MIN_STRICT_FULL_ROWS if experimental else None
            ),
            "strict_full_rows": verified_strict_rows,
            "quality_thresholds_sha256": vetted_thresholds_sha256,
            # Backward-compatible training provenance.  Downstream FEA must
            # use the separately sealed execution revisions below.
            "solver_revision": training_solver_revision,
            "library_revision": training_library_revision,
            "training_solver_revision": training_solver_revision,
            "training_library_revision": training_library_revision,
            "fea_solver_revision": fea_solver_revision,
            "fea_library_revision": fea_library_revision,
            "registry_generation": os.path.abspath(generation_artifact_root),
            "generation_report_sha256": _sha256(os.path.join(
                generation_artifact_root, "train_report.json"
            )),
            "model_artifacts_sha256": model_artifacts,
            "quality_status_sha256": (
                _sha256(supplied_quality) if supplied_quality else None
            ),
            "profile_sha256": (
                generation_report.get("profile_sha256")
                if args.registry_generation else None
            ),
            "pareto_front_sha256": _sha256(front_path),
            "pareto_X_sha256": _sha256(pareto_x_path),
            "pareto_F_sha256": _sha256(pareto_f_path),
            "sobol_schema": sobol_schema,
            "sobol_schema_sha256": sobol_schema_sha256,
            "required_models": REQUIRED_MODEL_TARGETS,
            "restarts": args.restarts,
            "population": args.pop,
            "workers": min(args.workers, args.restarts),
            "surrogate_inference_parallelism": inference_parallelism,
            "round": args.round,
            "seeds": [1000 + index for index in range(args.restarts)],
            "termination": {
                "ftol": 0.0025, "period": 30, "max_generations": 600,
            },
            "effective_spec": spec,
            "fixed_thermal_stack_mm": NSGA_FIXED_THERMAL_STACK_MM,
            "reported_flux_density_basis": (
                "B_design_analytic_T=V1_rms/"
                "(4*freq*N1*Ae_effective_m2)"
            ),
            "reported_flux_density_waveform": "bipolar_square",
            "reported_flux_density_denominator_coefficient": 4.0,
            "flux_density_constraint": (
                "B_design_analytic_T <= effective_spec.B_limit_T"
            ),
            "B_max_core_role": "diagnostic_only_not_an_optimization_constraint",
            "manufacturing_tolerance_policy": (
                "excluded; exact-as-FEA geometry is assumed"
            ),
        }, handle, indent=1)

    # This is the commit marker consumed by generation workers.  It is always
    # written after arrays, CSV, and the complete provenance manifest.
    completed_path = os.path.join(out_dir, "COMPLETED")
    staged_completed = completed_path + f".{os.getpid()}.tmp"
    with open(staged_completed, "w", encoding="utf-8") as handle:
        json.dump({
            "optimization_manifest_sha256": _sha256(manifest_path),
        }, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged_completed, completed_path)

    print(f"\nmerged Pareto: {len(Xp)} pts -> {out_dir}")
    print(f"volume {Fp[:,0].min():.0f}~{Fp[:,0].max():.0f} L | loss {Fp[:,1].min():.0f}~{Fp[:,1].max():.0f} W")


if __name__ == "__main__":
    main()
