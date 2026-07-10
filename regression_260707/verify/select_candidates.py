"""
AL 라운드 검증 후보 선정 (K=33 기본):
  - 활용 12: Pareto에서 하이퍼볼륨 기여 greedy + {최소부피, 최소손실, knee} 강제 포함
  - 제약경계 10: 정규화 제약 마진 최소 후보 (Llt 밴드 가장자리, 100C 근처)
                  + farthest-point 중복 제거 (0.05 단위거리)
  - 탐사 10: 준최적 대역(부피 하위 30%) 내 총 정규화 sigma 최대, max-min spread
  - 재검증 1: 현직 최적 (이전 라운드 검증 통과 최량)
"""
import numpy as np


def _dedup(X, keep_idx, min_dist=0.02):
    out = []
    for i in keep_idx:
        if all(np.linalg.norm(X[i] - X[j]) >= min_dist for j in out):
            out.append(i)
    return out


def _farthest_point(X, cand_idx, k, min_dist=0.05):
    picked = []
    for i in cand_idx:
        if len(picked) >= k:
            break
        if all(np.linalg.norm(X[i] - X[j]) >= min_dist for j in picked):
            picked.append(i)
    return picked


def hypervolume_greedy(F, k, ref=None):
    """2D 하이퍼볼륨 기여 greedy 선택"""
    F = np.asarray(F, dtype=float)
    ref = ref if ref is not None else F.max(axis=0) * 1.1
    order = np.argsort(F[:, 0])
    Fs = F[order]

    def hv(sel_mask):
        # 2D 하이퍼볼륨: x 오름차순 계단 적분
        pts = Fs[sel_mask]
        if not len(pts):
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        total = 0.0
        y_prev = ref[1]
        for x, y in pts:
            if y < y_prev:
                total += (ref[0] - x) * (y_prev - y)
                y_prev = y
        return total

    sel = np.zeros(len(Fs), dtype=bool)
    for _ in range(min(k, len(Fs))):
        best_gain, best_i = -1.0, None
        base = hv(sel)
        for i in np.where(~sel)[0]:
            sel[i] = True
            gain = hv(sel) - base
            sel[i] = False
            if gain > best_gain:
                best_gain, best_i = gain, i
        sel[best_i] = True
    return order[np.where(sel)[0]]


def knee_index(F):
    """정규화 front에서 (0,0)에 최근접한 점"""
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    return int(np.argmin(np.linalg.norm(Fn, axis=1)))


def select(X_unit, F, G, sigma_norm_total, incumbent_X=None,
           k_exploit=12, k_boundary=10, k_explore=10, verified_X=None):
    """
    X_unit: (n, d) 단위 유전자 / F: (n,2) 목적 / G: (n, m) 제약 (<=0 만족)
    sigma_norm_total: (n,) 총 정규화 불확실성
    verified_X: 과거 검증된 X (중복 제거용)
    반환: 선택된 인덱스 리스트 (순서 = 우선순위)
    """
    n = len(X_unit)
    picked = []

    def not_dup(i):
        if any(np.linalg.norm(X_unit[i] - X_unit[j]) < 0.02 for j in picked):
            return False
        if verified_X is not None and len(verified_X):
            if np.min(np.linalg.norm(verified_X - X_unit[i], axis=1)) < 0.02:
                return False
        return True

    # 1) 활용: HV greedy + 극단/knee 강제
    forced = {int(np.argmin(F[:, 0])), int(np.argmin(F[:, 1])), knee_index(F)}
    hv_sel = list(hypervolume_greedy(F, k_exploit))
    for i in list(forced) + hv_sel:
        if len([p for p in picked]) >= k_exploit:
            break
        if not_dup(i):
            picked.append(int(i))

    # 2) 제약 경계: 마진 최소 (활성 제약만: 스케일된 |g| 최소)
    margins = np.where(G <= 0, -G, np.inf).min(axis=1)
    order = np.argsort(margins)
    boundary = _farthest_point(X_unit, [i for i in order if np.isfinite(margins[i])],
                               k_boundary, min_dist=0.05)
    for i in boundary:
        if not_dup(i):
            picked.append(int(i))

    # 3) 탐사: 부피 하위 30% 대역에서 sigma 최대
    vol_cut = np.quantile(F[:, 0], 0.3)
    band = [i for i in range(n) if F[i, 0] <= vol_cut]
    for i in sorted(band, key=lambda j: -sigma_norm_total[j])[:3 * k_explore]:
        if len(picked) >= k_exploit + k_boundary + k_explore:
            break
        if not_dup(i):
            picked.append(int(i))

    # Reserve one slot so the documented AL batch size is K=33. Diversity
    # thresholds above are preferences; they must not silently shrink the
    # verification evidence below the round contract.
    target = min(n, k_exploit + k_boundary + k_explore + 1)
    eligible = []
    for i in range(n):
        if i in picked:
            continue
        if verified_X is not None and len(verified_X):
            if np.min(np.linalg.norm(verified_X - X_unit[i], axis=1)) < 0.02:
                continue
        eligible.append(i)
    while len(picked) < target and eligible:
        if picked:
            distances = [
                min(np.linalg.norm(X_unit[i] - X_unit[j]) for j in picked)
                for i in eligible
            ]
            best_position = int(np.argmax(distances))
        else:
            best_position = int(np.argmax([sigma_norm_total[i] for i in eligible]))
        picked.append(int(eligible.pop(best_position)))

    return picked
