"""Wild cluster bootstrap intervals for few, very unequal clusters.

The intact-cluster bootstrap redraws whole clusters, so with 12 repositories
where one holds 46% of the tasks, some replicates contain that repository
three times and others not at all: the spread of the draws mixes cluster
sizes with cluster outcomes. The wild cluster bootstrap keeps every cluster
in every replicate and instead flips the sign of each cluster's residual
sum (Rademacher weights). It is the standard remedy in econometrics for few
or unequal clusters (Cameron, Gelbach & Miller 2008; MacKinnon & Webb 2017).

Two variants, both studentized with the CR1 cluster-robust standard error:

  WCU  unrestricted residuals; one set of draws gives a bootstrap-t interval.
  WCR  residuals restricted to the null theta = theta0; the interval is the
       set of theta0 not rejected at 5%, found on a grid (test inversion).
       MacKinnon & Webb recommend WCR for unequal clusters.

With G <= 14 clusters all 2^G sign vectors are enumerated, so the bootstrap
is exact; otherwise they are sampled. Statistics are linear in per-task
values z (accuracy, accuracy gaps, and median cost through its influence
function), so each replicate costs one matrix product.

Marginal intervals only; frontier and dominance probabilities need joint
draws and stay with the intact-cluster bootstrap.
"""

from __future__ import annotations

import numpy as np

from .bootstrap import t_quantile
from claimeval.core.schema import RolloutSet


_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _weights(G: int, rng: np.random.Generator, n_max: int = 1 << 14) -> np.ndarray:
    """Sign vectors, one row per replicate.

    Below 10 clusters Rademacher weights allow too few distinct replicates
    (64 for G = 6) and the test turns conservative; Webb's six-point weights
    are the standard remedy (Webb 2023). Enumerated when feasible.
    """
    if G < 10:
        if 6 ** G <= 60_000:
            return _WEBB[np.indices((6,) * G).reshape(G, -1).T]
        return rng.choice(_WEBB, size=(n_max, G))
    if 2 ** G <= n_max:
        k = np.arange(2 ** G)[:, None]
        return 1.0 - 2.0 * ((k >> np.arange(G)[None, :]) & 1)
    return rng.choice((-1.0, 1.0), size=(n_max, G))


def _cluster_sums(z: np.ndarray, groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([z[g].sum() for g in groups])
    n = np.array([len(g) for g in groups], float)
    return A, n


def _cr1_var(sums: np.ndarray, n: np.ndarray, d: np.ndarray, T: float, G: int) -> np.ndarray:
    """CR1 variance of a mean; sums: (..., G) cluster sums of z - theta, d: mean of z - theta."""
    e = sums - n * d[..., None]
    return G / (G - 1) * (e ** 2).sum(axis=-1) / T ** 2


def wcu_interval(z: np.ndarray, groups, rng, level: float = 0.95) -> tuple[float, float]:
    """Unrestricted wild cluster bootstrap-t interval for mean(z)."""
    A, n = _cluster_sums(z, groups)
    T, G = n.sum(), len(n)
    th = A.sum() / T
    se = np.sqrt(_cr1_var(A - n * th, n, np.zeros(()), T, G))
    W = _weights(G, rng)
    S = W * (A - n * th)                     # (B, G) bootstrap residual sums
    d = S.sum(axis=1) / T
    t = d / np.sqrt(np.maximum(_cr1_var(S, n, d, T, G), 1e-300))
    lo_q, hi_q = np.quantile(t, [(1 - level) / 2, 1 - (1 - level) / 2])
    return th - hi_q * se, th - lo_q * se


def wcr_interval(z: np.ndarray, groups, rng, level: float = 0.95,
                 span: float = 12.0, iters: int = 30) -> tuple[float, float]:
    """Restricted wild cluster bootstrap interval for mean(z), by test inversion.

    Each endpoint is found by bisection on theta0 between the estimate (never
    rejected) and estimate +- span standard errors.
    """
    A, n = _cluster_sums(z, groups)
    T, G = n.sum(), len(n)
    th = A.sum() / T
    se = np.sqrt(_cr1_var(A - n * th, n, np.zeros(()), T, G))
    if not se > 0:
        return th, th
    W = _weights(G, rng)

    def rejected(theta0: float) -> bool:
        S = W * (A - n * theta0)
        d = S.sum(axis=1) / T
        t = d / np.sqrt(np.maximum(_cr1_var(S, n, d, T, G), 1e-300))
        t_obs = (th - theta0) / se
        return np.mean(np.abs(t) >= abs(t_obs) - 1e-12) < 1 - level

    out = []
    for side in (-1, 1):
        inner, outer = 0.0, span
        if not rejected(th + side * outer * se):
            out.append(th + side * outer * se)
            continue
        for _ in range(iters):
            mid = (inner + outer) / 2
            if rejected(th + side * mid * se):
                outer = mid
            else:
                inner = mid
        out.append(th + side * outer * se)
    return out[0], out[1]


def cr1_t_interval(z: np.ndarray, groups, level: float = 0.95) -> tuple[float, float]:
    """Analytic reference: CR1 standard error with t_{G-1} quantiles."""
    A, n = _cluster_sums(z, groups)
    T, G = n.sum(), len(n)
    th = A.sum() / T
    se = np.sqrt(_cr1_var(A - n * th, n, np.zeros(()), T, G))
    q = t_quantile(1 - (1 - level) / 2, G - 1)
    return th - q * se, th + q * se


def wild_intervals(rs: RolloutSet, variant: str = "wcr", level: float = 0.95,
                   rng: np.random.Generator | None = None, cost_stat: str = "median") -> dict:
    """Per-configuration wild cluster bootstrap intervals for accuracy and cost."""
    from claimeval.analysis.baselines import _cost_influence, _task_means

    rng = rng or np.random.default_rng(0)
    groups = [np.asarray(g) for g in rs.group_tasks]
    acc_t, cost_t = _task_means(rs)
    est_c, psi = _cost_influence(cost_t, cost_stat)
    f = {"wcr": wcr_interval, "wcu": wcu_interval}[variant]
    out = {k: np.empty(rs.n_configs) for k in ("acc_lo", "acc_hi", "cost_lo", "cost_hi")}
    for c in range(rs.n_configs):
        out["acc_lo"][c], out["acc_hi"][c] = f(np.nan_to_num(acc_t[c]), groups, rng, level)
        out["cost_lo"][c], out["cost_hi"][c] = f(est_c[c] + np.nan_to_num(psi[c]), groups, rng, level)
    return out
