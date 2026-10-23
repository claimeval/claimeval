"""Normalized rollout schema.

Everything downstream consumes a stream of `Rollout` tuples. That is the whole
portability story: HAL, pareto-evals, or an in-house runner only need to emit
(config, task, seed, success, cost) and the estimator applies unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Rollout:
    """One agent run of one task under one configuration."""

    config: str  # model x harness x effort x ... ; whatever you are comparing
    task: str
    seed: str  # anything distinguishing repeat runs of the same (config, task)
    success: float  # 1.0/0.0, or a continuous score in [0, 1]
    cost: float  # USD, or any positive resource measure (see cost_field in ingest)
    group: str = ""  # cluster the task belongs to: repo, domain, source dataset

    def __post_init__(self):
        # no group given -> each task is its own cluster, and group-level
        # resampling degenerates to task-level resampling
        if not self.group:
            object.__setattr__(self, "group", self.task)


class RolloutSet:
    """Dense, NaN-padded view over rollouts: (n_config, n_task, max_seeds).

    The padding matters. Real agent evaluations are ragged — configs crash on
    some tasks, reruns are uneven — and silently dropping those cells biases
    accuracy upward for exactly the configs that crash most.
    """

    def __init__(self, rollouts: Iterable[Rollout]):
        rollouts = list(rollouts)
        if not rollouts:
            raise ValueError("no rollouts")

        self.configs: list[str] = sorted({r.config for r in rollouts})
        self.tasks: list[str] = sorted({r.task for r in rollouts})
        ci = {c: i for i, c in enumerate(self.configs)}
        ti = {t: i for i, t in enumerate(self.tasks)}

        # task -> cluster. Real suites nest tasks in repos, domains or source
        # datasets; resampling tasks as if they were independent then
        # under-states variance (see bootstrap.hierarchical_bootstrap).
        task_group: dict[str, str] = {}
        for r in rollouts:
            prev = task_group.setdefault(r.task, r.group)
            if prev != r.group:
                raise ValueError(f"task {r.task!r} assigned to groups {prev!r} and {r.group!r}")
        self.groups: list[str] = sorted(set(task_group.values()))
        gi = {g: i for i, g in enumerate(self.groups)}
        self.task_group = np.array([gi[task_group[t]] for t in self.tasks], dtype=np.int64)
        self.group_tasks = [
            np.flatnonzero(self.task_group == g) for g in range(len(self.groups))
        ]

        cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
        seen: set[tuple[str, str, str]] = set()
        dupes = 0
        for r in rollouts:
            key = (r.config, r.task, r.seed)
            if key in seen:
                dupes += 1
            seen.add(key)
            cells.setdefault((ci[r.config], ti[r.task]), []).append((r.success, r.cost))
        if dupes:
            warnings.warn(
                f"{dupes} duplicate (config, task, seed) keys; kept as distinct seeds",
                stacklevel=2,
            )

        n_c, n_t = len(self.configs), len(self.tasks)
        self.counts = np.zeros((n_c, n_t), dtype=np.int64)
        for (c, t), vals in cells.items():
            self.counts[c, t] = len(vals)
        self.max_seeds = int(self.counts.max())

        self.success = np.full((n_c, n_t, self.max_seeds), np.nan)
        self.cost = np.full((n_c, n_t, self.max_seeds), np.nan)
        for (c, t), vals in cells.items():
            arr = np.asarray(vals, dtype=float)
            self.success[c, t, : len(vals)] = arr[:, 0]
            self.cost[c, t, : len(vals)] = arr[:, 1]

        if np.any(self.cost[~np.isnan(self.cost)] < 0):
            raise ValueError("negative costs")

    # -- shape ---------------------------------------------------------------

    @property
    def n_configs(self) -> int:
        return len(self.configs)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def is_clustered(self) -> bool:
        """True when tasks actually nest inside groups (not one task per group)."""
        return self.n_groups < self.n_tasks

    def coverage(self) -> np.ndarray:
        """Fraction of tasks each config actually ran. Anything < 1.0 makes the
        cross-config comparison unbalanced and should be reported, not hidden."""
        return (self.counts > 0).mean(axis=1)

    def seeds_per_cell(self) -> np.ndarray:
        """Median seeds per non-empty cell, per config."""
        out = np.zeros(self.n_configs)
        for c in range(self.n_configs):
            nz = self.counts[c][self.counts[c] > 0]
            out[c] = float(np.median(nz)) if nz.size else 0.0
        return out

    # -- point estimates -----------------------------------------------------

    def point_estimates(self, cost_stat: str = "median") -> tuple[np.ndarray, np.ndarray]:
        """(accuracy, cost) per config: aggregate over seeds, then over tasks.

        Mean-of-task-means, not mean-of-all-rollouts. With a ragged design the
        two differ, and only the former estimates "performance on a task drawn
        from the suite" — which is the estimand the frontier is about.

        `cost_stat` controls only the across-task step. The default is
        "median" because mean cost is not always a well-defined estimand: when
        cost scales exponentially with task difficulty and difficulty has
        polynomial tails, the population mean is infinite — and a finite sample
        will happily report a stable-looking finite number anyway. The median
        of per-task expected cost always exists. Use "mean" when you need
        total spend, knowing it is tail-dominated.
        """
        if cost_stat not in ("mean", "median"):
            raise ValueError(f"cost_stat must be 'mean' or 'median', got {cost_stat!r}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
            acc_t = np.nanmean(self.success, axis=2)
            cost_t = np.nanmean(self.cost, axis=2)
            agg = np.nanmean if cost_stat == "mean" else np.nanmedian
            return np.nanmean(acc_t, axis=1), agg(cost_t, axis=1)

    def cost_per_success(self) -> np.ndarray:
        """USD per *successful* task — the axis practitioners actually budget on.

        Infinite where a config never succeeds, which is the honest answer.
        """
        acc, cost = self.point_estimates()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(acc > 0, cost / acc, np.inf)

    def subsample_seeds(self, k: int, rng: np.random.Generator) -> "RolloutSet":
        """Draw k seeds per cell without replacement (for seed-count curves)."""
        out: list[Rollout] = []
        for c in range(self.n_configs):
            for t in range(self.n_tasks):
                n = int(self.counts[c, t])
                if n == 0:
                    continue
                pick = rng.permutation(n)[: min(k, n)]
                for s in pick:
                    out.append(
                        Rollout(
                            config=self.configs[c],
                            task=self.tasks[t],
                            seed=str(s),
                            success=float(self.success[c, t, s]),
                            cost=float(self.cost[c, t, s]),
                            group=self.groups[self.task_group[t]],
                        )
                    )
        return RolloutSet(out)

    def describe(self) -> str:
        acc, cost = self.point_estimates()
        cov, spc = self.coverage(), self.seeds_per_cell()
        lines = [
            f"{self.n_configs} configs x {self.n_tasks} tasks, "
            f"{int(np.nansum(self.counts))} rollouts",
            "",
            f"{'config':<34} {'acc':>7} {'cost':>10} {'seeds':>6} {'cover':>6}",
        ]
        for i, c in enumerate(self.configs):
            lines.append(
                f"{c[:34]:<34} {acc[i]:>7.3f} {cost[i]:>10.4f} "
                f"{spc[i]:>6.0f} {cov[i]:>6.0%}"
            )
        return "\n".join(lines)


def from_records(records: Sequence[dict]) -> RolloutSet:
    return RolloutSet(Rollout(**r) for r in records)
