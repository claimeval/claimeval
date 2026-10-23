"""Hierarchical bootstrap over the two nested sources of variance.

Agent evaluations have two: which tasks are in the suite, and how a given
(config, task) pair happened to go on a given run. Current practice reports
neither, or reports one and calls it the other. Separating them is what tells
you whether to buy more tasks or more seeds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .frontier import log_cost_range, pareto_mask
from claimeval.core.schema import RolloutSet

Mode = Literal["both", "task", "seed", "group", "cluster", "cluster_t", "cluster_cr2", "cluster_kish"]

_MAX_CELLS = 20_000_000  # chunk the replicate axis to bound peak memory


@dataclass
class BootstrapResult:
    acc: np.ndarray  # (B, n_config)
    cost: np.ndarray  # (B, n_config)
    configs: list[str]
    mode: Mode

    @property
    def n_replicates(self) -> int:
        return self.acc.shape[0]

    def frontier_prob(self) -> np.ndarray:
        """P(config is on the true frontier), the replacement for a binary flag."""
        return pareto_mask(self.cost, self.acc).mean(axis=0)

    def dominance_prob(self) -> np.ndarray:
        """P[i, j] = P(config i dominates config j)."""
        from .frontier import dominance

        return dominance(self.cost, self.acc).mean(axis=0)

    def ci(self, level: float = 0.95) -> dict[str, np.ndarray]:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return {
            "acc_lo": np.quantile(self.acc, lo, axis=0),
            "acc_hi": np.quantile(self.acc, hi, axis=0),
            "cost_lo": np.quantile(self.cost, lo, axis=0),
            "cost_hi": np.quantile(self.cost, hi, axis=0),
        }

    def sd(self) -> tuple[np.ndarray, np.ndarray]:
        return self.acc.std(axis=0, ddof=1), self.cost.std(axis=0, ddof=1)

    def log_cost_ref(self) -> tuple[float, float]:
        return log_cost_range(self.cost)


def hierarchical_bootstrap(
    rs: RolloutSet,
    n_boot: int = 2000,
    mode: Mode = "task",
    rng: np.random.Generator | None = None,
    cost_stat: str = "median",
) -> BootstrapResult:
    """Resample tasks and/or seeds.

    **Use mode="task" for inference.** Resampling tasks alone already captures
    total uncertainty, because each resampled cell carries its own seed noise
    through its observed mean. Additionally resampling seeds ("both") counts
    that within-cell component twice and over-states the variance.

    The coverage study makes this concrete: "both" covers 0.977 at a nominal
    0.95 with 3-5 seeds per cell, decaying to 0.960 at 20 seeds — the telltale
    1/S signature of the duplicated component — while "task" holds 0.937-0.948
    across the same range. Run `python -m claimeval coverage` to reproduce.

    **Use mode="cluster_t" when tasks are clustered** (SWE-bench instances nest in
    repos, GAIA questions in domains) and you want to generalize to new
    clusters. Task-level resampling then under-states variance, because tasks
    within a cluster share difficulty and are not independent draws. `auto_mode`
    picks this for you.

    "seed" isolates the within-cell component and exists for the variance
    decomposition, not for intervals.

    Task indices are resampled *once per replicate and shared across configs* —
    the configs are evaluated on a common suite, so treating their task draws as
    independent would inflate the variance of the difference between two
    configs, which is the quantity the frontier is made of.
    """
    rng = rng or np.random.default_rng(0)
    n_c, n_t, max_s = rs.n_configs, rs.n_tasks, rs.max_seeds
    valid_tasks = None  # (B, L) mask; only needed for ragged group draws

    if mode == "cluster_cr2":
        return _cluster_cr2(rs, n_boot, rng, cost_stat)
    if mode in ("cluster_t", "cluster_kish"):
        # intact-cluster resampling, widened for having few clusters; the Kish
        # variant takes degrees of freedom from the effective number of clusters
        b = hierarchical_bootstrap(rs, n_boot, "cluster", rng, cost_stat)
        acc0, cost0 = rs.point_estimates(cost_stat)
        k = few_cluster_scale(rs.n_groups)
        if mode == "cluster_kish":
            n = np.array([len(g) for g in rs.group_tasks], float)
            g_eff = max(n.sum() ** 2 / (n ** 2).sum(), 2.0)
            k *= t_quantile(0.975, max(int(round(g_eff - 1)), 1)) / t_quantile(0.975, rs.n_groups - 1)
        return BootstrapResult(acc0 + (b.acc - acc0) * k, cost0 + (b.cost - cost0) * k,
                               b.configs, "cluster_t")
    if mode in ("group", "cluster"):
        task_idx, valid_tasks = _group_task_indices(rs, n_boot, rng,
                                                    within=(mode == "group"))
    elif mode in ("both", "task"):
        task_idx = rng.integers(0, n_t, size=(n_boot, n_t))
    else:
        task_idx = np.tile(np.arange(n_t), (n_boot, 1))

    n_draw = task_idx.shape[1]
    acc_out = np.empty((n_boot, n_c))
    cost_out = np.empty((n_boot, n_c))
    resample_seeds = mode in ("both", "seed")
    chunk = max(1, min(n_boot, _MAX_CELLS // max(n_draw * max_s, 1)))
    arange_s = np.arange(max_s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN task cells
        for c in range(n_c):
            for start in range(0, n_boot, chunk):
                sl = slice(start, min(start + chunk, n_boot))
                ti = task_idx[sl]  # (b, n_draw)
                s_blk = rs.success[c][ti]  # (b, n_draw, max_s)
                k_blk = rs.cost[c][ti]
                counts = rs.counts[c][ti]  # (b, n_draw)

                if resample_seeds:
                    b = ti.shape[0]
                    draws = np.floor(
                        rng.random((b, n_draw, max_s)) * np.maximum(counts, 1)[..., None]
                    ).astype(np.int64)
                    np.clip(draws, 0, max_s - 1, out=draws)
                    s_blk = np.take_along_axis(s_blk, draws, axis=2)
                    k_blk = np.take_along_axis(k_blk, draws, axis=2)
                    valid = arange_s[None, None, :] < counts[..., None]
                    s_blk = np.where(valid, s_blk, np.nan)
                    k_blk = np.where(valid, k_blk, np.nan)

                acc_t = np.nanmean(s_blk, axis=2)  # (b, n_draw)
                cost_t = np.nanmean(k_blk, axis=2)
                if valid_tasks is not None:  # ragged group draws: mask the padding
                    vt = valid_tasks[sl]
                    acc_t = np.where(vt, acc_t, np.nan)
                    cost_t = np.where(vt, cost_t, np.nan)

                acc_out[sl, c] = np.nanmean(acc_t, axis=1)
                cost_agg = np.nanmean if cost_stat == "mean" else np.nanmedian
                cost_out[sl, c] = cost_agg(cost_t, axis=1)

    return BootstrapResult(acc_out, cost_out, list(rs.configs), mode)


def _group_task_indices(
    rs: RolloutSet, n_boot: int, rng: np.random.Generator, within: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Cluster bootstrap: resample whole groups, then tasks within each.

    Two stages, because both levels are sampled in reality — you observe some
    repos and some instances within them. Groups vary in size, so replicates
    vary in total length; the result is padded to the longest and returned with
    a validity mask.
    """
    n_g = rs.n_groups
    sizes = np.array([len(t) for t in rs.group_tasks])
    picks = rng.integers(0, n_g, size=(n_boot, n_g))  # (B, n_g) group draws
    lengths = sizes[picks].sum(axis=1)
    max_len = int(lengths.max())

    task_idx = np.zeros((n_boot, max_len), dtype=np.int64)
    valid = np.zeros((n_boot, max_len), dtype=bool)
    for b in range(n_boot):
        pos = 0
        for g in picks[b]:
            tasks = rs.group_tasks[g]
            k = len(tasks)
            # within=False keeps each drawn cluster intact (Field & Welsh's
            # "cluster bootstrap"); within=True also resamples its tasks
            # ("two-stage"), which can count within-cluster noise twice
            task_idx[b, pos : pos + k] = (rng.choice(tasks, size=k, replace=True)
                                          if within else tasks)
            pos += k
        valid[b, :pos] = True
    return task_idx, (valid if not valid.all() else None)


def t_quantile(p: float, dof: int) -> float:
    """Student-t quantile by bisection on a numerically integrated CDF (no scipy)."""
    from math import gamma, pi, sqrt
    c = gamma((dof + 1) / 2) / (sqrt(dof * pi) * gamma(dof / 2))
    xs = np.linspace(0, 60, 600001)
    dens = c * (1 + xs ** 2 / dof) ** (-(dof + 1) / 2)
    cdf = 0.5 + np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(xs))])
    return float(np.interp(p, cdf, xs))


_SCALE_CACHE: dict[int, float] = {}


def few_cluster_scale(G: int) -> float:
    """Widening for an intact-cluster bootstrap over G clusters.

    sqrt(G/(G-1)) undoes the bootstrap's plug-in variance shrinkage, and
    t_{G-1}/z swaps normal for Student-t quantiles, as cluster-robust
    inference does with few clusters (Cameron, Gelbach & Miller 2008).
    """
    if G not in _SCALE_CACHE:
        _SCALE_CACHE[G] = float(np.sqrt(G / (G - 1)) * t_quantile(0.975, G - 1)
                                / 1.959963984540054)
    return _SCALE_CACHE[G]


def _cr2_variance(z: np.ndarray, rs: RolloutSet) -> float:
    """Leverage-corrected cluster-robust variance of the mean of z (one value per task).

    CR2 (Bell & McCaffrey 2002) for an intercept-only model: each cluster's
    centred sum is inflated by 1/(1 - h_g), h_g = n_g / T, the cluster's
    leverage on the overall mean. A cluster holding much of the data pulls the
    mean toward itself, so its residual understates its true deviation.
    """
    T = len(z)
    v = 0.0
    for idx in rs.group_tasks:
        h = len(idx) / T
        v += z[idx].sum() ** 2 / max(1.0 - h, 1e-6)
    return v / T ** 2


def _cluster_cr2(rs: RolloutSet, n_boot: int, rng: np.random.Generator,
                 cost_stat: str) -> BootstrapResult:
    """Intact-cluster bootstrap rescaled to the CR2 variance, with t_{G-1} quantiles.

    Each statistic's bootstrap spread is matched to its CR2 variance, so the
    joint shape of the draws (correlations between configurations, which the
    frontier depends on) comes from the bootstrap, and only the scale from CR2.
    With equal cluster sizes CR2 reduces to the G/(G-1) correction.
    """
    from claimeval.analysis.baselines import _cost_influence, _task_means

    b = hierarchical_bootstrap(rs, n_boot, "cluster", rng, cost_stat)
    acc0, cost0 = rs.point_estimates(cost_stat)
    acc_t, cost_t = _task_means(rs)
    _, psi = _cost_influence(cost_t, cost_stat)
    tq = t_quantile(0.975, rs.n_groups - 1) / 1.959963984540054
    acc = np.empty_like(b.acc)
    cost = np.empty_like(b.cost)
    for c in range(rs.n_configs):
        for out, draws, est, z in ((acc, b.acc, acc0, np.nan_to_num(acc_t[c] - acc0[c])),
                                   (cost, b.cost, cost0, np.nan_to_num(psi[c]))):
            sd_boot = draws[:, c].std(ddof=1)
            sd_cr2 = np.sqrt(_cr2_variance(z, rs))
            k = (sd_cr2 / sd_boot if sd_boot > 0 else 1.0) * tq
            out[:, c] = est[c] + (draws[:, c] - est[c]) * k
    return BootstrapResult(acc, cost, b.configs, "cluster_cr2")


def auto_mode(rs: RolloutSet) -> Mode:
    """Pick the right resampling unit: clusters if the suite has them, else tasks."""
    return "cluster_t" if rs.is_clustered else "task"


def variance_decomposition(
    rs: RolloutSet, n_boot: int = 1000, rng: np.random.Generator | None = None
) -> dict[str, np.ndarray]:
    """Split total accuracy variance into between-task and within-task parts.

    The task-only bootstrap estimates *total* variance (see
    `hierarchical_bootstrap`), and the seed-only bootstrap estimates the
    within-cell component. So the between-task component is the difference,
    not the task-only figure itself — adding the two would double-count.

    `task_share` -> 1 means variance is dominated by which tasks are in the
    suite: buy more tasks. -> 0 means it is dominated by run-to-run
    stochasticity: buy more seeds. This is the hand-off into adaptive budget
    allocation.
    """
    rng = rng or np.random.default_rng(0)
    v_total = hierarchical_bootstrap(rs, n_boot, "task", rng).acc.var(axis=0, ddof=1)
    v_within = hierarchical_bootstrap(rs, n_boot, "seed", rng).acc.var(axis=0, ddof=1)
    # clipped at 0: the two estimates are independent, so their difference can
    # go slightly negative when the within component dominates
    v_between = np.maximum(v_total - v_within, 0.0)
    return {
        "sd_total": np.sqrt(v_total),
        "sd_task": np.sqrt(v_between),
        "sd_seed": np.sqrt(v_within),
        "task_share": np.divide(
            v_between, v_total, out=np.full_like(v_total, np.nan), where=v_total > 0
        ),
    }
