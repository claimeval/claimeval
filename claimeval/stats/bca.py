"""BCa (bias-corrected and accelerated) bootstrap intervals for per-configuration
accuracy and cost (Efron 1987; Efron & Tibshirani 1993, ch. 14).

Percentile intervals assume the bootstrap distribution is centred and
symmetric about the truth. With few tasks, or a skewed statistic such as mean
cost when costs are log-normal across tasks, neither holds and coverage drops.
BCa adjusts the percentile endpoints using
  z0 -- the bias: how far the bootstrap distribution sits from the estimate,
  a  -- the acceleration: skewness of the statistic, from a jackknife over tasks.

Marginal intervals only. Frontier and dominance probabilities stay as they are.
"""

from __future__ import annotations

import warnings

import numpy as np

from .bootstrap import hierarchical_bootstrap
from claimeval.core.schema import RolloutSet


def norm_cdf(x):
    x = np.asarray(x, float)
    z = np.abs(x) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    erf = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027
                    + t * 1.061405429)))) * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * erf)


def norm_ppf(p):
    """Inverse normal CDF (Acklam's rational approximation, |rel error| < 1.2e-9)."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    m = p < lo
    q = np.sqrt(-2 * np.log(p[m]))
    out[m] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5
    r = q * q
    out[m] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
             (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    m = p > hi
    q = np.sqrt(-2 * np.log(1 - p[m]))
    out[m] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return out


def _jackknife(values: np.ndarray, agg) -> np.ndarray:
    """Leave-one-task-out statistics. values: (C, T) per-task values."""
    C, T = values.shape
    out = np.empty((C, T))
    for t in range(T):
        keep = np.ones(T, bool)
        keep[t] = False
        out[:, t] = agg(values[:, keep], axis=1)
    return out


def bca_intervals(rs: RolloutSet, n_boot: int = 2000, level: float = 0.95,
                  rng: np.random.Generator | None = None, cost_stat: str = "median") -> dict:
    """BCa intervals for accuracy and cost per configuration (task resampling)."""
    rng = rng or np.random.default_rng(0)
    b = hierarchical_bootstrap(rs, n_boot, "task", rng, cost_stat)
    acc0, cost0 = rs.point_estimates(cost_stat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        acc_t = np.nanmean(rs.success, axis=2)
        cost_t = np.nanmean(rs.cost, axis=2)
    agg_c = np.nanmean if cost_stat == "mean" else np.nanmedian
    out = {}
    for name, draws, est, vals, agg in (("acc", b.acc, acc0, acc_t, np.nanmean),
                                         ("cost", b.cost, cost0, cost_t, agg_c)):
        jk = _jackknife(vals, agg)
        lo, hi = np.empty(len(est)), np.empty(len(est))
        for c in range(len(est)):
            d = draws[:, c]
            prop = (np.sum(d < est[c]) + 0.5 * np.sum(d == est[c])) / len(d)
            z0 = norm_ppf(prop)
            u = jk[c].mean() - jk[c]
            den = 6 * (np.sum(u ** 2) ** 1.5)
            a = np.sum(u ** 3) / den if den > 0 else 0.0
            zs = norm_ppf([(1 - level) / 2, 1 - (1 - level) / 2])
            adj = norm_cdf(z0 + (z0 + zs) / np.maximum(1 - a * (z0 + zs), 1e-9))
            lo[c], hi[c] = np.quantile(d, adj)
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out
