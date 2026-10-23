"""Calibrated simulator for agent rollouts.

Reproduces the three properties that break off-the-shelf multi-objective
statistics, so the estimator can be tested against known ground truth before
any real spend:

1. success and cost are dependent within a rollout, with the sign flipping by
   failure mode (fail-fast is cheap, max-turn-loop is expensive);
2. cost is heavy-tailed with a point mass at the turn cap;
3. task difficulty is a shared latent, so config errors are correlated.

Split into `make_population` (config parameters, held fixed) and
`sample_dataset` (tasks and seeds, redrawn). Coverage studies need exactly that
separation: the estimand is the population value for a *fixed* set of configs,
with the task suite as the thing being resampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from claimeval.core.schema import Rollout


@dataclass(frozen=True)
class Population:
    """Ground-truth generative parameters for a set of configs."""

    ability: np.ndarray  # logit-scale skill
    log_cost_base: np.ndarray  # log10 USD for a typical solved task
    cap: np.ndarray  # max-turn cost ceiling (the point mass)
    difficulty_sd: float = 1.0
    grind_frac: float = 0.6  # share of failures that grind to the cap
    bail_mult: float = 0.35  # cost multiplier for fail-fast runs
    cost_sd: float = 0.30
    diff_cost_coef: float = 0.30  # harder tasks cost more even when solved

    # --- robustness knobs: task-difficulty distribution ---------------------
    difficulty_dist: str = "normal"  # "normal" | "heavy" | "clustered"
    n_clusters: int = 10
    icc: float = 0.6  # share of difficulty variance that is between-cluster
    # Real suites have unequal clusters (SWE-bench Verified: django is 231 of
    # 500 tasks). When set, overrides n_clusters and fixes the suite size.
    cluster_sizes: tuple | None = None
    heavy_df: float = 3.0
    partial_credit: float = 0.0  # >0: graded scores with this concentration

    @property
    def n_configs(self) -> int:
        return len(self.ability)

    def names(self) -> list[str]:
        # zero-padded so lexicographic order (used by RolloutSet) matches index
        return [f"cfg{c:02d}_a{a:+.1f}" for c, a in enumerate(self.ability)]


def make_population(n_configs: int = 9, rng: np.random.Generator | None = None) -> Population:
    """Ability and log-cost correlated but not perfectly.

    That imperfect correlation is the point: the population frontier then holds
    both genuinely Pareto-optimal configs and near-misses that noise can
    promote, which is the regime the estimator has to survive.
    """
    rng = rng or np.random.default_rng(0)
    ability = np.linspace(-0.8, 1.6, n_configs)
    log_cost_base = 0.55 * ability + rng.normal(0, 0.25, n_configs) - 1.0
    return Population(
        ability=ability,
        log_cost_base=log_cost_base,
        cap=10 ** (log_cost_base + 0.85),
    )


def _draw_difficulty_marginal(pop: Population, n: int, rng: np.random.Generator) -> np.ndarray:
    """Marginal task-difficulty distribution — what the estimand integrates over.

    Clustering changes the *dependence* between tasks in a suite, not this
    marginal, so the population truth is identical across the normal and
    clustered families. That is what makes the clustered case a clean test of
    the variance estimate alone.
    """
    if pop.difficulty_dist == "heavy":
        scale = pop.difficulty_sd / np.sqrt(pop.heavy_df / (pop.heavy_df - 2))
        return rng.standard_t(pop.heavy_df, n) * scale
    return rng.normal(0, pop.difficulty_sd, n)


def _draw_suite(
    pop: Population, n_tasks: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[str]]:
    """Difficulties plus the cluster label of each task."""
    if pop.difficulty_dist != "clustered":
        return _draw_difficulty_marginal(pop, n_tasks, rng), [f"g{t:04d}" for t in range(n_tasks)]

    between = pop.difficulty_sd * np.sqrt(pop.icc)
    within = pop.difficulty_sd * np.sqrt(1 - pop.icc)
    if pop.cluster_sizes is not None:
        if sum(pop.cluster_sizes) != n_tasks:
            raise ValueError(f"cluster_sizes sum to {sum(pop.cluster_sizes)}, "
                             f"but n_tasks={n_tasks}")
        assign = np.repeat(np.arange(len(pop.cluster_sizes)), pop.cluster_sizes)
        means = rng.normal(0, between, len(pop.cluster_sizes))
    else:
        means = rng.normal(0, between, pop.n_clusters)
        assign = np.arange(n_tasks) % pop.n_clusters  # balanced clusters
    diff = means[assign] + rng.normal(0, within, n_tasks)
    return diff, [f"clust{g:02d}" for g in assign]


def sample_dataset(
    pop: Population,
    n_tasks: int = 50,
    n_seeds: int = 20,
    rng: np.random.Generator | None = None,
) -> list[Rollout]:
    """Draw a fresh task suite and fresh rollouts from a fixed population."""
    rng = rng or np.random.default_rng(0)
    names = pop.names()
    difficulty, groups = _draw_suite(pop, n_tasks, rng)
    p = 1 / (1 + np.exp(-(pop.ability[:, None] - difficulty[None, :])))

    rolls: list[Rollout] = []
    for c in range(pop.n_configs):
        for t in range(n_tasks):
            solved = rng.random(n_seeds) < p[c, t]  # drives cost either way
            if pop.partial_credit > 0:
                # graded score with the same mean p, so the estimand is
                # unchanged and only the estimator's input distribution differs
                k = pop.partial_credit
                succ = rng.beta(max(p[c, t] * k, 1e-3), max((1 - p[c, t]) * k, 1e-3), n_seeds)
            else:
                succ = solved.astype(float)
            mu = pop.log_cost_base[c] + pop.diff_cost_coef * difficulty[t]
            cost = 10 ** rng.normal(mu, pop.cost_sd, n_seeds)
            fail = ~solved
            grind = fail & (rng.random(n_seeds) < pop.grind_frac)
            cost[fail & ~grind] *= pop.bail_mult
            cost[grind] = pop.cap[c]
            for s in range(n_seeds):
                rolls.append(
                    Rollout(
                        names[c], f"task{t:03d}", str(s),
                        float(succ[s]), float(cost[s]), groups[t],
                    )
                )
    return rolls


def population_truth(
    pop: Population,
    n_mc: int = 400_000,
    rng: np.random.Generator | None = None,
    cost_stat: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Population (accuracy, cost) per config: the estimand the bootstrap targets.

    Conditional on task difficulty both objectives have closed forms, so the
    only Monte Carlo is over the difficulty distribution. That makes truth far
    more precise than any simulated dataset, which matters — MC error in the
    reference value would show up as spurious undercoverage.

    `test_population_truth_matches_sampler` checks this agrees with the actual
    sampler, so the closed form cannot silently drift from the code path the
    estimator sees.
    """
    rng = rng or np.random.default_rng(12345)
    d = _draw_difficulty_marginal(pop, n_mc, rng)  # (n_mc,)
    L = np.log(10.0)

    p = 1 / (1 + np.exp(-(pop.ability[:, None] - d[None, :])))  # (C, n_mc)
    mu = pop.log_cost_base[:, None] + pop.diff_cost_coef * d[None, :]
    # E[10^N(mu, sd)] for the lognormal solved-task cost
    solved = np.exp(L * mu + (L * pop.cost_sd) ** 2 / 2)
    failed = pop.grind_frac * pop.cap[:, None] + (1 - pop.grind_frac) * pop.bail_mult * solved
    cond_cost = p * solved + (1 - p) * failed  # E[cost | difficulty]
    agg = np.mean if cost_stat == "mean" else np.median
    return p.mean(axis=1), agg(cond_cost, axis=1)


def simulate(
    n_configs: int = 9,
    n_tasks: int = 50,
    n_seeds: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[list[Rollout], dict]:
    """Convenience wrapper: one population, one dataset, plus population truth."""
    rng = rng or np.random.default_rng(0)
    pop = make_population(n_configs, rng)
    rolls = sample_dataset(pop, n_tasks, n_seeds, rng)
    acc, cost = population_truth(pop, n_mc=100_000)
    return rolls, {
        "configs": pop.names(),
        "accuracy": acc,
        "cost": cost,
        "ability": pop.ability,
        "pop": pop,
    }
