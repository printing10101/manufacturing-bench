"""统计显著性检验（stats）。

基于 exp15 + 四阶段实验的统计实践：
- welch_t: 独立样本 t（方差不齐时）
- paired_t: 配对 t（同组跨 seed 对比，如外推 vs 域内）
- cohens_d: 效应量（d>0.8 为大效应）
- mean_ci: 均值 95% 置信区间
- hierarchical_diff: 层级贝叶斯近似的组差异后验（正态-逆伽马共轭）
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import stats


def _as_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def welch_t(a: Sequence[float], b: Sequence[float]) -> dict:
    """Welch 独立样本 t 检验。返回 t, p, dof, 效应量 d。"""
    a, b = _as_array(a), _as_array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p), "d": cohens_d(a, b),
            "n1": int(len(a)), "n2": int(len(b))}


def paired_t(a: Sequence[float], b: Sequence[float]) -> dict:
    """配对 t 检验（同一组实验的两个条件，如外推 vs 域内）。"""
    a, b = _as_array(a), _as_array(b)
    if len(a) != len(b):
        raise ValueError("paired t 要求等长")
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "d": cohens_d(a - b, np.zeros_like(a)),
            "n": int(len(a))}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d（池化标准差效应量）。"""
    a, b = _as_array(a), _as_array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def mean_ci(a: Sequence[float], confidence: float = 0.95) -> dict:
    """均值置信区间。"""
    a = _as_array(a)
    n = len(a)
    if n < 2:
        return {"mean": float(a.mean()), "ci_low": float(a.mean()),
                "ci_high": float(a.mean()), "n": n}
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return {"mean": float(m), "ci_low": float(m - h), "ci_high": float(m + h), "n": n}


def hierarchical_diff(a: Sequence[float], b: Sequence[float],
                      n_samples: int = 20000) -> dict:
    """层级贝叶斯近似：组差异后验 P(μ_a > μ_b)。

    正态-逆伽马共轭（组内正态 + Jeffreys 先验），解析采样组均值后验。
    返回：P(a>b)、后验均值差、95% HDI 近似（分位数）。
    """
    a, b = _as_array(a), _as_array(b)

    def _posterior_samples(x: np.ndarray, n: int) -> np.ndarray:
        x = x[~np.isnan(x)]
        m, n_ = x.mean(), len(x)
        if n_ < 2:
            return np.full(n, m)
        v = x.var(ddof=1)
        # 逆伽马后验（Jeffreys）：shape=(n_-1)/2, scale=(n_-1)*v/2
        sig2 = 1.0 / np.random.gamma((n_ - 1) / 2, 2.0 / ((n_ - 1) * v), size=n)
        # 组均值的条件正态
        mu = np.random.normal(m, np.sqrt(sig2 / n_))
        return mu

    rng_state = np.random.get_state()
    np.random.seed(0)  # 可复现后验采样
    try:
        ma = _posterior_samples(a, n_samples)
        mb = _posterior_samples(b, n_samples)
    finally:
        np.random.set_state(rng_state)

    diff = ma - mb
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "p_a_gt_b": float(np.mean(diff > 0)),
        "mean_diff": float(diff.mean()),
        "hdi_low": float(lo),
        "hdi_high": float(hi),
    }
