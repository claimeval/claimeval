"""Pareto frontier geometry: dominance, membership, Hausdorff distance, bands."""

from __future__ import annotations

import numpy as np


def dominance(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Boolean tensor D[..., i, j] = "config i dominates config j".

    Minimize cost, maximize accuracy. Weak in both, strict in at least one.
    Identical points therefore do not dominate each other, so exact ties are
    both retained rather than both eliminated.

    Batched over leading axes, so this runs on all bootstrap replicates at once.
    """
    le = cost[..., :, None] <= cost[..., None, :]
    ge = acc[..., :, None] >= acc[..., None, :]
    strict = (cost[..., :, None] < cost[..., None, :]) | (
        acc[..., :, None] > acc[..., None, :]
    )
    return le & ge & strict


def pareto_mask(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated configs, batched over leading axes."""
    nan = np.isnan(cost) | np.isnan(acc)
    d = dominance(np.nan_to_num(cost, nan=np.inf), np.nan_to_num(acc, nan=-np.inf))
    return ~d.any(axis=-2) & ~nan


def _normalize(cost: np.ndarray, acc: np.ndarray, ref: tuple[float, float]):
    """Map to a comparable [0,1]^2 space before measuring distance.

    Cost is log-scaled because agent costs span orders of magnitude and the
    meaningful difference is multiplicative: $1 -> $2 matters as much as
    $100 -> $200. Using raw dollars would make the band width a function of
    whichever expensive config happens to be in the study.
    """
    lo, hi = ref
    lc = np.log10(np.maximum(cost, 1e-12))
    span = max(hi - lo, 1e-12)
    return np.stack([(lc - lo) / span, acc], axis=-1)


def log_cost_range(cost: np.ndarray) -> tuple[float, float]:
    finite = cost[np.isfinite(cost) & (cost > 0)]
    if finite.size == 0:
        return (0.0, 1.0)
    lc = np.log10(finite)
    return (float(lc.min()), float(lc.max()))


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two finite point sets in R^2."""
    if a.size == 0 or b.size == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def frontier_points(
    cost: np.ndarray, acc: np.ndarray, ref: tuple[float, float]
) -> np.ndarray:
    """Normalized points of the non-dominated set, sorted along cost."""
    m = pareto_mask(cost, acc)
    pts = _normalize(cost[m], acc[m], ref)
    return pts[np.argsort(pts[:, 0])] if pts.size else pts


def frontier_band(
    boot_cost: np.ndarray,
    boot_acc: np.ndarray,
    ref: tuple[float, float],
    level: float = 0.90,
    n_candidates: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    """Confidence band for the frontier *as a curve*, not per-point intervals.

    Follows the Bassi et al. construction: find the most representative member
    of the family of bootstrap fronts (the one minimizing mean Hausdorff
    distance to the others), then take the `level` quantile of distances to it
    as the band radius.

    Exact selection of the median front is O(B^2) Hausdorff computations, so the
    candidate pool is subsampled to `n_candidates`. Distances of all B fronts to
    the chosen median are computed exactly — the approximation affects only
    which front is called central, not the reported radius.
    """
    rng = rng or np.random.default_rng(0)
    B = boot_cost.shape[0]
    fronts = [frontier_points(boot_cost[b], boot_acc[b], ref) for b in range(B)]

    cand_idx = rng.choice(B, size=min(n_candidates, B), replace=False)
    cands = [fronts[i] for i in cand_idx]
    mean_d = [np.mean([hausdorff(c, o) for o in cands]) for c in cands]
    median_front = cands[int(np.argmin(mean_d))]

    dists = np.array([hausdorff(f, median_front) for f in fronts])
    return {
        "median_front": median_front,
        "radius": float(np.quantile(dists, level)),
        "distances": dists,
        "level": level,
    }
