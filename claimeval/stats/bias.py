"""Selection bias on frontier members, and the seed-count curve.

A config lands on the reported frontier partly because it is good and partly
because it got lucky. Conditioning on selection makes the point estimate
optimistic — the winner's curse. This module measures it.
"""

from __future__ import annotations

import numpy as np

from .frontier import frontier_points, hausdorff, log_cost_range, pareto_mask
from claimeval.core.schema import RolloutSet


def split_sample_debias(
    rs: RolloutSet, n_splits: int = 200, rng: np.random.Generator | None = None
) -> dict:
    """Estimate winner's-curse bias by selecting and estimating on disjoint seeds.

    Split the seeds of every cell in half. Choose the frontier on half A; read
    off accuracy and cost from half B. Because B played no part in selection,
    the A-minus-B gap for selected configs is an unbiased estimate of the
    inflation induced by selection. Averaged over random splits.

    Requires >= 2 seeds per cell. With single-seed data this is not estimable,
    and the function says so rather than returning a number that looks like one.
    """
    rng = rng or np.random.default_rng(0)
    usable = rs.counts[rs.counts > 0]
    if usable.size == 0 or usable.min() < 2:
        return {
            "available": False,
            "reason": "needs >= 2 seeds per cell; single-seed runs cannot "
            "separate selection from estimation",
        }

    acc_gap, cost_gap, n_sel = [], [], []
    for _ in range(n_splits):
        a, b = _split_halves(rs, rng)
        acc_a, cost_a = a.point_estimates()
        acc_b, cost_b = b.point_estimates()
        sel = pareto_mask(cost_a, acc_a)
        if not sel.any():
            continue
        acc_gap.append(np.nanmean(acc_a[sel] - acc_b[sel]))
        cost_gap.append(np.nanmean(cost_a[sel] - cost_b[sel]))
        n_sel.append(int(sel.sum()))

    acc_gap = np.asarray(acc_gap)
    cost_gap = np.asarray(cost_gap)
    acc_pt, cost_pt = rs.point_estimates()
    on = pareto_mask(cost_pt, acc_pt)
    mean_acc_on = float(np.nanmean(acc_pt[on])) if on.any() else np.nan

    return {
        "available": True,
        "acc_bias": float(acc_gap.mean()),
        "acc_bias_se": float(acc_gap.std(ddof=1) / np.sqrt(len(acc_gap))),
        "cost_bias": float(cost_gap.mean()),
        "rel_acc_bias": float(acc_gap.mean() / mean_acc_on) if mean_acc_on else np.nan,
        "mean_frontier_size": float(np.mean(n_sel)),
        "n_splits": len(acc_gap),
    }


def _split_halves(rs: RolloutSet, rng: np.random.Generator):
    """Partition each cell's seeds into two disjoint RolloutSets."""
    from claimeval.core.schema import Rollout

    left: list[Rollout] = []
    right: list[Rollout] = []
    for c in range(rs.n_configs):
        for t in range(rs.n_tasks):
            n = int(rs.counts[c, t])
            if n < 2:
                continue
            perm = rng.permutation(n)
            half = n // 2
            for bucket, idx in ((left, perm[:half]), (right, perm[half : 2 * half])):
                for s in idx:
                    bucket.append(
                        Rollout(
                            rs.configs[c],
                            rs.tasks[t],
                            str(s),
                            float(rs.success[c, t, s]),
                            float(rs.cost[c, t, s]),
                            rs.groups[rs.task_group[t]],
                        )
                    )
    return RolloutSet(left), RolloutSet(right)


def seed_count_curve(
    rs: RolloutSet,
    ks: tuple[int, ...] = (1, 2, 3, 5, 10),
    n_repeats: int = 40,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """How wrong is a k-seed frontier, against the full-data frontier?

    This is the E2 analysis, runnable on any dataset deep enough in seeds. The
    reference is the full-seed estimate, which is itself noisy — so read the
    numbers as "error relative to a much better estimate", not absolute truth.
    """
    rng = rng or np.random.default_rng(0)
    acc_ref, cost_ref = rs.point_estimates()
    ref_mask = pareto_mask(cost_ref, acc_ref)
    ref_range = log_cost_range(cost_ref)
    ref_front = frontier_points(cost_ref, acc_ref, ref_range)
    max_k = int(rs.counts.max())

    out = []
    for k in ks:
        if k > max_k:
            continue
        jaccard, dists, acc_infl, cost_infl, sizes = [], [], [], [], []
        for _ in range(n_repeats):
            sub = rs.subsample_seeds(k, rng)
            acc_k, cost_k = sub.point_estimates()
            mask_k = pareto_mask(cost_k, acc_k)
            union = (mask_k | ref_mask).sum()
            jaccard.append((mask_k & ref_mask).sum() / union if union else np.nan)
            dists.append(hausdorff(frontier_points(cost_k, acc_k, ref_range), ref_front))
            sizes.append(int(mask_k.sum()))
            if mask_k.any():
                # inflation = what k seeds claims for its winners, minus what the
                # full data says about those same configs
                acc_infl.append(np.nanmean(acc_k[mask_k] - acc_ref[mask_k]))
                cost_infl.append(np.nanmean(cost_k[mask_k] - cost_ref[mask_k]))
        out.append(
            {
                "k": k,
                "jaccard_vs_full": float(np.nanmean(jaccard)),
                "hausdorff_vs_full": float(np.nanmean(dists)),
                "acc_inflation": float(np.nanmean(acc_infl)),
                "cost_inflation": float(np.nanmean(cost_infl)),
                "frontier_size": float(np.mean(sizes)),
            }
        )
    return out
