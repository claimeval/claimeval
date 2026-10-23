"""Test the kinds of claims papers and leaderboards actually make.

A claim names a subject configuration and, usually, comparators:

  top1            subject is the most accurate of {subject} + comparators
  pairwise        subject is more accurate than each comparator
  margin          subject beats each comparator by at least `margin` (accuracy, 0-1)
  cost_dominance  subject is at least as accurate and no more costly than each comparator
  frontier        subject is on the cost-accuracy Pareto frontier of the given set
  topk            at most k-1 comparators are more accurate than subject ("#k", "top 5")
  equivalent      subject is within `margin` of each comparator ("as good as", "on par")
  cheaper_noninferior  subject costs less than each comparator and is no more than
                  `margin` less accurate ("as good as, but cheaper")

Each claim gets a bootstrap probability. It is *supported* at level `thr`
(default 0.9) when that probability reaches it -- the same certification rule
used throughout the paper. "Not supported" means the data cannot distinguish
the claim from its negation at that confidence; it does not mean the claim is
false.

    from claimeval import RolloutSet
    from claimeval.claims import Claim, evaluate_claims
    res = evaluate_claims(rs, [Claim("top1", "agentA", ["agentB", "agentC"])])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from claimeval.stats.bootstrap import Mode, auto_mode, hierarchical_bootstrap
from claimeval.stats.frontier import pareto_mask
from claimeval.core.schema import RolloutSet

KINDS = ("top1", "pairwise", "margin", "cost_dominance", "frontier", "topk", "equivalent",
         "cheaper_noninferior")


@dataclass
class Claim:
    kind: str
    subject: str
    comparators: list[str] = field(default_factory=list)
    margin: float = 0.0
    id: str = ""
    k: int = 1

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown claim kind {self.kind!r}; expected one of {KINDS}")


def claim_probability(acc: np.ndarray, cost: np.ndarray, claim: Claim, idx: dict) -> np.ndarray:
    """Per-replicate truth value of the claim. acc, cost: (B, n_configs)."""
    s = idx[claim.subject]
    comp = [idx[c] for c in claim.comparators]
    a, c = acc[:, s], cost[:, s]
    if claim.kind == "top1":
        return np.all(a[:, None] > acc[:, comp], axis=1)
    if claim.kind == "pairwise":
        return np.all(a[:, None] > acc[:, comp], axis=1)
    if claim.kind == "margin":
        return np.all(a[:, None] - acc[:, comp] >= claim.margin, axis=1)
    if claim.kind == "topk":
        return (acc[:, comp] > a[:, None]).sum(axis=1) <= claim.k - 1
    if claim.kind == "equivalent":
        return np.all(np.abs(a[:, None] - acc[:, comp]) <= claim.margin, axis=1)
    if claim.kind == "cheaper_noninferior":
        return np.all((c[:, None] < cost[:, comp]) & (a[:, None] >= acc[:, comp] - claim.margin), axis=1)
    if claim.kind == "cost_dominance":
        return np.all((a[:, None] >= acc[:, comp]) & (c[:, None] <= cost[:, comp]), axis=1)
    keep = [s] + comp
    return pareto_mask(cost[:, keep], acc[:, keep])[:, 0]


def evaluate_claims(rs: RolloutSet, claims: list[Claim], n_boot: int = 4000, mode: Mode | None = None,
                rng: np.random.Generator | None = None, thr: float = 0.9,
                cost_stat: str = "median") -> list[dict]:
    """Bootstrap probability and verdict for each claim, from one shared set of draws."""
    rng = rng or np.random.default_rng(0)
    mode = mode or auto_mode(rs)
    b = hierarchical_bootstrap(rs, n_boot, mode, rng, cost_stat)
    acc0, cost0 = rs.point_estimates(cost_stat)
    idx = {k: i for i, k in enumerate(rs.configs)}
    out = []
    for cl in claims:
        missing = [k for k in [cl.subject, *cl.comparators] if k not in idx]
        if missing:
            raise KeyError(f"claim {cl.id or cl.subject}: unknown configs {missing}")
        p = float(claim_probability(b.acc, b.cost, cl, idx).mean())
        pt = bool(claim_probability(acc0[None], cost0[None], cl, idx)[0])
        out.append({"id": cl.id, "kind": cl.kind, "subject": cl.subject,
                    "comparators": list(cl.comparators), "probability": p,
                    "holds_at_point_estimate": pt, "supported": p >= thr, "mode": mode,
                    "subject_acc": float(acc0[idx[cl.subject]])})
    return out
