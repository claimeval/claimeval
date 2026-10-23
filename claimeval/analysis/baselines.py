"""Existing uncertainty methods, in the same output format as ours.

Every method returns a BootstrapResult: B joint draws of (accuracy, cost) for
all configurations. Frontier-membership and dominance probabilities, intervals
and certified claims are then computed by the SAME code for every method, so a
comparison measures the error model and nothing else.

Methods (paper names in brackets):

  naive          [current practice]  point estimates, no uncertainty.
  wald           [per-model error bars] independent normal intervals per
                 configuration, tasks treated as independent. The +/- usually
                 printed next to a leaderboard score.
  paired_clt     [Miller 2024, paired] joint normal over all configurations
                 with the full task-level covariance, so comparisons are
                 paired. Tasks independent.
  clustered_clt  [Miller 2024, clustered] as paired_clt with cluster-robust
                 (sandwich) covariance over task groups, small-sample
                 corrected. The strongest formula-based competitor.
  boot_both      [naive bootstrap] resample tasks and runs together.
  boot_task          [ours, unclustered] task resampling.
  boot_cluster_t     [ours, clustered] intact clusters + few-cluster widening.
  boot_group         [our earlier version] two-stage; double-counts within-cluster noise.
  boot_cluster       [ablation] intact clusters, no widening.

The formula methods have no native notion of P(on frontier). They get the
fairest extension available: draws from their own sampling distribution,
pushed through the same frontier code (a parametric bootstrap).

Median cost has no simple standard error. The formula methods use its
influence function, psi = (1/2 - 1{x <= m}) / f(m), with a Gaussian kernel
density estimate of f at the median -- the textbook delta-method route.
"""

from __future__ import annotations

import warnings

import numpy as np

from claimeval.stats.bootstrap import BootstrapResult, hierarchical_bootstrap
from claimeval.core.schema import RolloutSet

METHODS = ("naive", "wald", "paired_clt", "clustered_clt", "boot_both", "boot_task", "boot_group",
           "boot_cluster", "boot_cluster_t", "boot_cluster_cr2", "boot_cluster_kish")


def _task_means(rs: RolloutSet) -> tuple[np.ndarray, np.ndarray]:
    """(C, T) per-task mean success and cost, averaged over runs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(rs.success, axis=2), np.nanmean(rs.cost, axis=2)


def _cost_influence(cost_t: np.ndarray, stat: str) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and per-task influence values for the cost statistic.

    Returns (est (C,), psi (C, T)) such that est is approximately the mean of
    est + psi, i.e. psi plays the role of centred per-task contributions.
    """
    if stat == "mean":
        est = np.nanmean(cost_t, axis=1)
        return est, cost_t - est[:, None]
    est = np.nanmedian(cost_t, axis=1)
    psi = np.zeros_like(cost_t)
    for c in range(cost_t.shape[0]):
        x = cost_t[c][~np.isnan(cost_t[c])]
        n = len(x)
        sd = np.std(x, ddof=1)
        iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
        h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * n ** -0.2  # Silverman
        h = max(h, 1e-12)
        f = np.mean(np.exp(-0.5 * ((x - est[c]) / h) ** 2)) / (h * np.sqrt(2 * np.pi))
        f = max(f, 1e-12)
        psi[c] = (0.5 - (cost_t[c] <= est[c])) / f
    return est, psi


def _normal_draws(mean: np.ndarray, cov: np.ndarray, B: int, rng) -> np.ndarray:
    # eigen-decomposition tolerates the rank deficiency of a covariance
    # estimated from fewer clusters than dimensions
    w, V = np.linalg.eigh((cov + cov.T) / 2)
    w = np.clip(w, 0, None)
    return mean + (rng.standard_normal((B, len(mean))) * np.sqrt(w)) @ V.T


def _formula(rs: RolloutSet, B: int, rng, cost_stat: str, kind: str) -> BootstrapResult:
    acc_t, cost_t = _task_means(rs)
    C, T = acc_t.shape
    acc = np.nanmean(acc_t, axis=1)
    cost_est, psi = _cost_influence(cost_t, cost_stat)
    # centred per-task contributions for all 2C statistics
    Z = np.vstack([acc_t - acc[:, None], psi])  # (2C, T)
    Z = np.nan_to_num(Z)
    mean = np.concatenate([acc, cost_est])

    if kind == "wald":
        var = (Z ** 2).sum(axis=1) / (T * (T - 1))
        cov = np.diag(var)  # every configuration independent
    elif kind == "paired_clt":
        cov = (Z @ Z.T) / (T * (T - 1))
    elif kind == "clustered_clt":
        G = rs.n_groups
        S = np.zeros((2 * C, G))
        for g, idx in enumerate(rs.group_tasks):
            S[:, g] = Z[:, idx].sum(axis=1)
        cov = (S @ S.T) / T ** 2 * (G / (G - 1))  # CR1 small-sample correction
    else:
        raise ValueError(kind)

    draws = _normal_draws(mean, cov, B, rng)
    return BootstrapResult(draws[:, :C], draws[:, C:], list(rs.configs), kind)


def run_method(rs: RolloutSet, method: str, B: int = 1000, rng=None,
               cost_stat: str = "median") -> BootstrapResult:
    rng = rng or np.random.default_rng(0)
    if method == "naive":
        acc, cost = rs.point_estimates(cost_stat)
        return BootstrapResult(acc[None, :], cost[None, :], list(rs.configs), "naive")
    if method in ("wald", "paired_clt", "clustered_clt"):
        if method == "clustered_clt" and not rs.is_clustered:
            method = "paired_clt"  # no groups: the clustered estimator is the paired one
        return _formula(rs, B, rng, cost_stat, method)
    if method in ("boot_cluster_cr2", "boot_cluster_kish"):
        mode = method[5:] if rs.is_clustered else "task"
        return hierarchical_bootstrap(rs, n_boot=B, mode=mode, rng=rng, cost_stat=cost_stat)
    if method == "boot_cluster_t":
        mode = "cluster_t" if rs.is_clustered else "task"
        return hierarchical_bootstrap(rs, n_boot=B, mode=mode, rng=rng, cost_stat=cost_stat)
    mode = {"boot_both": "both", "boot_task": "task", "boot_group": "group",
            "boot_cluster": "cluster"}[method]
    if mode in ("group", "cluster") and not rs.is_clustered:
        mode = "task"
    return hierarchical_bootstrap(rs, n_boot=B, mode=mode, rng=rng, cost_stat=cost_stat)
