"""
NSGA-2 멀티 재시작 드라이버.

- pop 200, ftol 수렴 종료 (고정 세대수 대신), 재시작 N회 (시드 다양화)
- warm start: 이전 라운드 Pareto(archive/pareto_X.npy)를 초기 개체군에 주입
- 전역 Pareto 병합 + 전체 평가 아카이브 (AL 인필 선정용)

사용:
  python run_nsga2.py --restarts 16 --round 1
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "training"))

from optimization.nsga2_problem import MFTProblem, T_TARGETS, DEFAULT_SPEC  # noqa: E402


def load_models(registry=None):
    from predictor import EnsemblePredictor
    import predictor as predictor_mod
    reg = registry or predictor_mod.REGISTRY
    targets = ["Llt_phys", "P_winding_total", "P_core_total", "P_core_plate_total",
               "B_max_core"] + T_TARGETS
    models = {}
    for t in targets:
        path = os.path.join(reg, t, "models.pkl")
        if os.path.isfile(path):
            models[t] = EnsemblePredictor.load(t, registry=reg)
    return models


def build_density_gate(dataset_path, features):
    from predictor import DensityGate
    from checkpoint_train import to_physical
    df = to_physical(pd.read_parquet(dataset_path))
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
    ap.add_argument("--spec", default=None, help="스펙 오버라이드 JSON")
    ap.add_argument("--no-density-gate", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "..", "al_rounds", f"round_{args.round:02d}")
    os.makedirs(out_dir, exist_ok=True)

    models = load_models()
    missing = [t for t in ["Llt_phys", "P_winding_total", "P_core_total", "B_max_core"]
               if t not in models]
    if missing:
        raise SystemExit(f"필수 모델 미학습: {missing} (train_models.py 먼저)")

    spec = dict(DEFAULT_SPEC)
    if args.spec:
        spec.update(json.load(open(args.spec)))

    gate = None
    if not args.no_density_gate:
        gate = build_density_gate(args.dataset, models["Llt_phys"].features)

    problem = MFTProblem(models, spec=spec, density_gate=gate)

    # warm start: 이전 라운드 Pareto
    warm = None
    prev = os.path.join(HERE, "..", "al_rounds", f"round_{args.round - 1:02d}", "pareto_X.npy")
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
    np.save(os.path.join(out_dir, "pareto_X.npy"), Xp)
    np.save(os.path.join(out_dir, "pareto_F.npy"), Fp)

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
    pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1) \
        .to_csv(os.path.join(out_dir, "pareto_front.csv"), index=False)

    print(f"\nmerged Pareto: {len(Xp)} pts -> {out_dir}")
    print(f"volume {Fp[:,0].min():.0f}~{Fp[:,0].max():.0f} L | loss {Fp[:,1].min():.0f}~{Fp[:,1].max():.0f} W")


if __name__ == "__main__":
    main()
