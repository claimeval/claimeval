"""Double-bootstrap recalibration of frontier-membership probabilities.

P(config on frontier) is the bootstrap frequency of a compound event: being
non-dominated by every other configuration at once. Such frequencies are
distorted -- in simulation, a forecast of 0.25 comes true about half as
often, and one of 0.85 more often -- because the bootstrap world is centred on
the estimates, not on the truth.

The double bootstrap measures that distortion on the data itself (Hall 1992,
ch. 3; Efron & Tibshirani 1993, ch. 18). In the bootstrap world the observed
data are the population, so "truth" is known: the frontier of the point
estimates. Each outer resample plays the role of a new dataset; an inner
bootstrap on it yields membership probabilities, which are scored against
that known truth. A monotone (isotonic) map from forecast to observed
frequency is fitted and applied to the original probabilities.

No model is assumed beyond the bootstrap itself; the price is B_outer inner
bootstraps.
"""

from __future__ import annotations

import copy

import numpy as np

from .bootstrap import Mode, auto_mode, hierarchical_bootstrap
from .frontier import pareto_mask
from claimeval.core.schema import RolloutSet


def subset(rs: RolloutSet, idx: np.ndarray) -> RolloutSet:
    """A RolloutSet holding tasks `idx` (repeats allowed, each repeat a new task)."""
    out = copy.copy(rs)
    out.tasks = [f"{rs.tasks[t]}#{k}" for k, t in enumerate(idx)]
    out.success, out.cost, out.counts = rs.success[:, idx], rs.cost[:, idx], rs.counts[:, idx]
    out.max_seeds = rs.max_seeds
    tg = rs.task_group[idx]
    out.task_group = tg
    out.groups = [str(g) for g in np.unique(tg)]
    out.group_tasks = [np.flatnonzero(tg == g) for g in np.unique(tg)]
    return out


def _outer_draw(rs: RolloutSet, mode: Mode, rng) -> RolloutSet:
    if mode in ("task", "both"):
        return subset(rs, rng.integers(0, rs.n_tasks, rs.n_tasks))
    # intact clusters; a cluster drawn twice counts as two clusters
    picks = rng.integers(0, rs.n_groups, rs.n_groups)
    out = subset(rs, np.concatenate([rs.group_tasks[g] for g in picks]))
    lab = np.repeat(np.arange(len(picks)), [len(rs.group_tasks[g]) for g in picks])
    out.task_group, out.groups = lab, [str(g) for g in range(len(picks))]
    out.group_tasks = [np.flatnonzero(lab == g) for g in range(len(picks))]
    return out


def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit; returns (knots, values) for np.interp."""
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o].astype(float)
    vals, wts, xs = [], [], []
    for xi, yi in zip(x, y):
        vals.append(yi)
        wts.append(1.0)
        xs.append([xi])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            xs[-2] += xs[-1]
            vals[-2:], wts[-2:] = [v], [w]
            xs.pop()
    knots = np.array([np.mean(b) for b in xs])
    return knots, np.array(vals)


def calibration_map(rs: RolloutSet, b_outer: int = 100, b_inner: int = 200,
                    mode: Mode | None = None, rng=None, cost_stat: str = "median"):
    """Fit the double-bootstrap map from P(frontier) forecast to frequency."""
    rng = rng or np.random.default_rng(0)
    mode = mode or auto_mode(rs)
    acc0, cost0 = rs.point_estimates(cost_stat)
    truth = pareto_mask(cost0, acc0).astype(float)
    preds, outs = [], []
    for _ in range(b_outer):
        rs_o = _outer_draw(rs, mode, rng)
        p = hierarchical_bootstrap(rs_o, b_inner, mode, rng, cost_stat).frontier_prob()
        preds.append(p)
        outs.append(truth)
    knots, vals = isotonic(np.concatenate(preds), np.concatenate(outs))
    return lambda p: np.clip(np.interp(p, knots, vals), 0.0, 1.0)


def calibrated_frontier_prob(rs: RolloutSet, n_boot: int = 2000, b_outer: int = 100,
                             b_inner: int = 200, mode: Mode | None = None, rng=None,
                             cost_stat: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """(raw, recalibrated) P(config on frontier)."""
    rng = rng or np.random.default_rng(0)
    mode = mode or auto_mode(rs)
    raw = hierarchical_bootstrap(rs, n_boot, mode, rng, cost_stat).frontier_prob()
    f = calibration_map(rs, b_outer, b_inner, mode, rng, cost_stat)
    return raw, f(raw)
