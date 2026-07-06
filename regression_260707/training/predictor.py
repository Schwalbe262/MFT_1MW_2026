"""
학습된 앙상블 로더 + NSGA-2 인터페이스 (predict_mu_sigma) + 데이터 밀도 게이트.
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry")

from checkpoint_train import inverse_y  # noqa: E402


class EnsemblePredictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.features = bundle["features"]
        self.kind = bundle["transform"]
        self.q90 = bundle["q90"]

    @classmethod
    def load(cls, target, registry=REGISTRY):
        with open(os.path.join(registry, target, "models.pkl"), "rb") as f:
            return cls(pickle.load(f))

    def predict_mu_sigma(self, X_df, conformal=True):
        """X_df: 특징 프레임 (여분 컬럼 무시). 반환: (mu, sigma) 원공간, sigma는 q90 보정 반폭."""
        X = X_df.reindex(columns=self.features).fillna(0.0)
        preds_t = np.stack([m.predict(X) for _, m in self.bundle["models"]])
        mu_t = np.median(preds_t, axis=0)
        sg_t = preds_t.std(axis=0)
        mu = inverse_y(mu_t, self.kind)
        deriv = np.abs(inverse_y(mu_t + 1e-4, self.kind) - inverse_y(mu_t - 1e-4, self.kind)) / 2e-4
        sg = np.maximum(deriv * sg_t, 1e-9)
        if conformal:
            sg = sg * self.q90
        return mu, sg

    def disagreement(self, X_df):
        """앙상블 불일치 (원공간 max-min 폭) - 신뢰 게이트용"""
        X = X_df.reindex(columns=self.features).fillna(0.0)
        preds_t = np.stack([m.predict(X) for _, m in self.bundle["models"]])
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
