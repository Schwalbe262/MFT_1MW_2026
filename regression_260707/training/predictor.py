"""
학습된 앙상블 로더 + NSGA-2 인터페이스 (predict_mu_sigma) + 데이터 밀도 게이트.
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry")
LEGACY_SIGMA_FLOOR_POLICY = "legacy_absolute_1e-9"
RELATIVE_SIGMA_FLOOR_POLICY = "relative_machine_epsilon_v1"

from checkpoint_train import inverse_y  # noqa: E402


class EnsemblePredictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.features = bundle["features"]
        self.kind = bundle["transform"]
        self.q90 = bundle["q90"]
        self.inference_threads = None
        self.sigma_floor_policy = bundle.get(
            "sigma_floor_policy", LEGACY_SIGMA_FLOOR_POLICY
        )
        if self.sigma_floor_policy not in {
            LEGACY_SIGMA_FLOOR_POLICY,
            RELATIVE_SIGMA_FLOOR_POLICY,
        }:
            raise RuntimeError(
                f"unsupported sigma floor policy: {self.sigma_floor_policy}"
            )

    def configure_inference_threads(self, threads=1):
        """Bound nested model prediction parallelism for outer-parallel jobs.

        NSGA-II already parallelizes independent restarts.  Letting every
        ensemble member also use all host cores creates nested joblib/OpenMP
        pools, repeated sklearn warnings and severe oversubscription.  Keep
        the default unchanged for other consumers; optimization explicitly
        opts into this bound after loading its pinned model generation.
        """

        if isinstance(threads, bool):
            raise ValueError("inference threads must be a positive integer")
        try:
            normalized = int(threads)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "inference threads must be a positive integer"
            ) from exc
        if normalized < 1 or normalized > 4:
            raise ValueError("inference threads must be between 1 and 4")
        supported = {"lightgbm", "xgboost", "catboost", "extratrees"}
        configured = []
        for family, model in self.bundle["models"]:
            family_name = str(family).lower()
            if family_name not in supported:
                raise RuntimeError(
                    f"cannot bound unsupported ensemble family: {family}"
                )
            if family_name != "catboost":
                if not hasattr(model, "n_jobs"):
                    raise RuntimeError(
                        f"{family_name} model has no n_jobs inference control"
                    )
                model.n_jobs = normalized
                if int(model.n_jobs) != normalized:
                    raise RuntimeError(
                        f"failed to bind {family_name} inference threads"
                    )
            configured.append(family_name)
        self.inference_threads = normalized
        return {
            "threads": normalized,
            "model_count": len(configured),
            "families": configured,
        }

    def _predict_model(self, family, model, frame):
        if (
            self.inference_threads is not None
            and str(family).lower() == "catboost"
        ):
            return model.predict(
                frame, thread_count=int(self.inference_threads)
            )
        return model.predict(frame)

    def _transformed_predictions(self, frame):
        return np.stack([
            self._predict_model(family, model, frame)
            for family, model in self.bundle["models"]
        ])

    @classmethod
    def load(cls, target, registry=REGISTRY):
        # There is intentionally no legacy flat-file fallback.  Only an atomic
        # schema-v2 pointer with matching passing gate evidence is loadable.
        from train_models import load_active_generation

        active = load_active_generation(registry)
        return cls._load_record(target, active)

    @classmethod
    def load_generation(cls, target, registry, generation):
        """Load an explicitly pinned generation, requiring accepted gate evidence."""
        from train_models import load_generation

        return cls._load_record(
            target, load_generation(registry, generation, require_accepted=True)
        )

    @classmethod
    def _load_record(cls, target, active):
        model_path = os.path.join(active["generation"], target, "models.pkl")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        if bundle.get("training_run_id") != active["report"].get("training_run_id"):
            raise RuntimeError("model bundle does not match active generation")
        if bundle.get("dataset_sha256") != active["report"].get("dataset_sha256"):
            raise RuntimeError("model bundle dataset does not match active generation")
        return cls(bundle)

    def predict_mu_sigma(self, X_df, conformal=True):
        """X_df: 특징 프레임 (여분 컬럼 무시). 반환: (mu, sigma) 원공간, sigma는 q90 보정 반폭."""
        X = X_df.reindex(columns=self.features).fillna(0.0)
        preds_t = self._transformed_predictions(X)
        mu_t = np.median(preds_t, axis=0)
        sg_t = preds_t.std(axis=0)
        mu = inverse_y(mu_t, self.kind)
        deriv = np.abs(inverse_y(mu_t + 1e-4, self.kind) - inverse_y(mu_t - 1e-4, self.kind)) / 2e-4
        if self.sigma_floor_policy == LEGACY_SIGMA_FLOOR_POLICY:
            sigma_floor = np.full_like(mu, 1e-9, dtype=float)
        else:
            sigma_floor = np.maximum(
                np.abs(mu) * np.finfo(float).eps, np.finfo(float).tiny
            )
        sg = np.maximum(deriv * sg_t, sigma_floor)
        if conformal:
            sg = sg * self.q90
        return mu, sg

    def disagreement(self, X_df):
        """앙상블 불일치 (원공간 max-min 폭) - 신뢰 게이트용"""
        X = X_df.reindex(columns=self.features).fillna(0.0)
        preds_t = self._transformed_predictions(X)
        lo = inverse_y(preds_t.min(axis=0), self.kind)
        hi = inverse_y(preds_t.max(axis=0), self.kind)
        return hi - lo


class DensityGate:
    """학습셋 k-NN 거리 기반 외삽 게이트: gate(X) > 0 이면 제약 위반 (데이터 희박 지역)"""

    def __init__(self, X_train_df, features, k=8, quantile=0.95):
        from sklearn.neighbors import NearestNeighbors
        self.features = features
        A = X_train_df.reindex(columns=features).fillna(0.0).to_numpy(dtype=float)
        self.mean = A.mean(axis=0)
        self.std = A.std(axis=0) + 1e-12
        Z = (A - self.mean) / self.std
        self.nn = NearestNeighbors(n_neighbors=k).fit(Z)
        d, _ = self.nn.kneighbors(Z)
        self.threshold = float(np.quantile(d.mean(axis=1), quantile))

    def __call__(self, X_df):
        Z = ((X_df.reindex(columns=self.features).fillna(0.0).to_numpy(dtype=float)
              - self.mean) / self.std)
        d, _ = self.nn.kneighbors(Z)
        return d.mean(axis=1) - self.threshold
