"""
NSGA-2 멀티 재시작 드라이버.

- pop 200, ftol 수렴 종료 (고정 세대수 대신), 재시작 N회 (시드 다양화)
- warm start: 이전 라운드 Pareto(archive/pareto_X.npy)를 초기 개체군에 주입
- 전역 Pareto 병합 + 전체 평가 아카이브 (AL 인필 선정용)

사용:
  python run_nsga2.py --restarts 16 --round 1
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "training"))

from optimization.nsga2_problem import MFTProblem, T_TARGETS, DEFAULT_SPEC  # noqa: E402

REQUIRED_MODEL_TARGETS = [
    "Llt_phys", "k",
    "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
    "B_max_core", "B_mean_core", *T_TARGETS,
]
MIN_STRICT_FULL_ROWS = 3000
VETTED_QUALITY_THRESHOLDS = os.path.abspath(os.path.join(
    HERE, "..", "training", "model_quality_thresholds.json"
))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recomputed_strict_full_rows(
        dataset_path, solver_revision=None, library_revision=None):
    from quality_contract import annotate_validity

    audited = annotate_validity(
        pd.read_parquet(dataset_path),
        expected_solver_revision=solver_revision,
        expected_library_revision=library_revision,
    )
    return int(audited["_strict_valid_full"].sum())


def load_models(registry=None):
    from predictor import EnsemblePredictor
    import predictor as predictor_mod
    reg = registry or predictor_mod.REGISTRY
    models = {}
    for t in REQUIRED_MODEL_TARGETS:
        try:
            models[t] = EnsemblePredictor.load(t, registry=reg)
        except (FileNotFoundError, OSError):
            pass
    return models


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

    algo = NSGA2(pop_size=pop, sampling=init)
    term = DefaultMultiObjectiveTermination(ftol=0.0025, period=30, n_max_gen=max_gen)
    res = minimize(problem, algo, term, seed=int(seed), verbose=False, save_history=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--round", type=int, default=0, help="AL 라운드 번호 (출력 디렉토리)")
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data", "dataset", "train.parquet"))
    ap.add_argument("--registry", default=os.path.join(HERE, "..", "training", "registry"))
    ap.add_argument("--registry-generation", default=None)
    ap.add_argument("--quality-status", default=None)
    ap.add_argument("--output-root", default=os.path.join(HERE, "..", "al_rounds"))
    ap.add_argument(
        "--quality-thresholds",
        default=VETTED_QUALITY_THRESHOLDS,
    )
    ap.add_argument("--spec", default=None, help="스펙 오버라이드 JSON")
    ap.add_argument("--no-density-gate", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_root, f"round_{args.round:02d}")
    os.makedirs(out_dir, exist_ok=True)

    vetted_thresholds_sha256 = _sha256(VETTED_QUALITY_THRESHOLDS)
    if _sha256(args.quality_thresholds) != vetted_thresholds_sha256:
        raise SystemExit("quality thresholds differ from the vetted production contract")

    if bool(args.registry_generation) != bool(args.quality_status):
        raise SystemExit(
            "--registry-generation and --quality-status must be supplied together"
        )
    registry_for_models = args.registry
    generation_report = {}
    if args.registry_generation:
        registry_for_models = os.path.abspath(args.registry_generation)
        with open(args.quality_status, encoding="utf-8") as handle:
            quality = json.load(handle)
        with open(
            os.path.join(registry_for_models, "train_report.json"), encoding="utf-8"
        ) as handle:
            generation_report = json.load(handle)
        dataset_sha = _sha256(args.dataset)
        if (not quality.get("passed")
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
        if (verified_strict_rows < MIN_STRICT_FULL_ROWS
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

    models = load_models(registry_for_models)
    missing = [target for target in REQUIRED_MODEL_TARGETS if target not in models]
    if missing:
        raise SystemExit(f"required model generation is incomplete: {missing}")

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
        warm = np.load(prev)
        print(f"warm start: {len(warm)} points from round {args.round - 1}")

    F_list, X_list, feasible_counts = [], [], []
    for r in range(args.restarts):
        res = run_one(problem, seed=1000 + r, pop=args.pop, warm_X=warm)
        if res.X is None:
            feasible_counts.append(0)
            continue
        X = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        X_list.append(X)
        F_list.append(F)
        feasible_counts.append(len(X))
        print(f"restart {r}: {len(X)} pareto pts, vol {F[:,0].min():.0f}-{F[:,0].max():.0f}L, "
              f"loss {F[:,1].min():.0f}-{F[:,1].max():.0f}W, gen {res.algorithm.n_gen}")

    if not X_list:
        raise SystemExit("모든 재시작에서 feasible 해 없음 - 제약 조임/데이터 커버리지 점검 필요")

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
        for t, mdl in models.items():
            mu, sg = mdl.predict_mu_sigma(frame.iloc[[i]])
            row[f"pred_{t}"] = float(mu[0])
            row[f"sigma_{t}"] = float(sg[0])
        rows.append(row)
    front_path = os.path.join(out_dir, "pareto_front.csv")
    pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1) \
        .to_csv(front_path, index=False)

    model_artifacts = {
        target: {
            name: _sha256(os.path.join(registry_for_models, target, name))
            for name in ("models.pkl", "meta.json")
        }
        for target in REQUIRED_MODEL_TARGETS
    }

    with open(os.path.join(out_dir, "optimization_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "training_run_id": quality.get("training_run_id"),
            "dataset_sha256": quality.get("dataset_sha256"),
            "quality_gate_passed": True,
            "strict_full_rows": verified_strict_rows,
            "quality_thresholds_sha256": vetted_thresholds_sha256,
            "solver_revision": quality.get("solver_revision"),
            "library_revision": quality.get("library_revision"),
            "registry_generation": os.path.abspath(registry_for_models),
            "generation_report_sha256": _sha256(os.path.join(
                registry_for_models, "train_report.json"
            )),
            "model_artifacts_sha256": model_artifacts,
            "quality_status_sha256": (
                _sha256(args.quality_status) if args.quality_status else None
            ),
            "profile_sha256": (
                generation_report.get("profile_sha256")
                if args.registry_generation else None
            ),
            "pareto_front_sha256": _sha256(front_path),
            "pareto_X_sha256": _sha256(pareto_x_path),
            "pareto_F_sha256": _sha256(pareto_f_path),
            "required_models": REQUIRED_MODEL_TARGETS,
            "restarts": args.restarts,
            "population": args.pop,
            "round": args.round,
            "seeds": [1000 + index for index in range(args.restarts)],
            "termination": {
                "ftol": 0.0025, "period": 30, "max_generations": 600,
            },
            "effective_spec": spec,
            "manufacturing_tolerance_policy": (
                "excluded; exact-as-FEA geometry is assumed"
            ),
        }, handle, indent=1)

    print(f"\nmerged Pareto: {len(Xp)} pts -> {out_dir}")
    print(f"volume {Fp[:,0].min():.0f}~{Fp[:,0].max():.0f} L | loss {Fp[:,1].min():.0f}~{Fp[:,1].max():.0f} W")


if __name__ == "__main__":
    main()
