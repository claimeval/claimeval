"""Head-to-head comparison of uncertainty methods on shared simulated data.

Each replication draws one dataset from a scenario with known truth; every
method sees that same dataset. Reported per method:

  acc_cov / cost_cov   95% interval coverage (target 0.95)
  acc_w / cost_w       mean interval width (coverage bought by width is no win)
  gap_cov              coverage of accuracy differences between adjacent configs
  dom_prec             share of certified dominance claims (P >= 0.9) that are true
  dom_n                certified dominance claims per replication (power)
  dom_ece              calibration error of P(i dominates j)
  front_prec/recall    certified frontier membership vs the true frontier

"naive" has no intervals; its claims are the point-estimate claims, asserted
with certainty, which is what current practice does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .baselines import run_method
from claimeval.sim.coverage import Calibration
from claimeval.stats.frontier import dominance, pareto_mask
from claimeval.core.schema import RolloutSet


@dataclass
class Scenario:
    name: str
    draw: Callable[[np.random.Generator], RolloutSet]  # one dataset
    true_acc: np.ndarray
    true_cost: np.ndarray
    cost_stat: str = "median"


def compare(scenario: Scenario, methods, reps: int = 200, B: int = 800,
            thr: float = 0.9, level: float = 0.95, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    ta, tc = scenario.true_acc, scenario.true_cost
    true_dom = dominance(tc, ta)
    np.fill_diagonal(true_dom, False)
    true_front = pareto_mask(tc, ta)
    order = np.argsort(ta)
    adj = list(zip(order[1:], order[:-1]))  # (better, worse) by true accuracy
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2

    acc_names = {m: {k: [] for k in ("acc_cov", "cost_cov", "acc_w", "cost_w", "gap_cov",
                                      "dom_true", "front_p", "front_r")} for m in methods}
    dom_n = {m: 0 for m in methods}
    cal = {m: Calibration() for m in methods}

    for _ in range(reps):
        rs = scenario.draw(rng)
        idx = {k: i for i, k in enumerate(rs.configs)}
        assert len(idx) == len(ta), "scenario configs must match truth"
        a_hat, c_hat = rs.point_estimates(scenario.cost_stat)
        naive_dom = dominance(c_hat, a_hat)
        np.fill_diagonal(naive_dom, False)
        naive_front = pareto_mask(c_hat, a_hat)
        for m in methods:
            b = run_method(rs, m, B=B, rng=rng, cost_stat=scenario.cost_stat)
            r = acc_names[m]
            if m != "naive":
                alo, ahi = np.quantile(b.acc, [lo_q, hi_q], axis=0)
                clo, chi = np.quantile(b.cost, [lo_q, hi_q], axis=0)
                r["acc_cov"].append(np.mean((alo <= ta) & (ta <= ahi)))
                r["cost_cov"].append(np.mean((clo <= tc) & (tc <= chi)))
                r["acc_w"].append(np.mean(ahi - alo))
                r["cost_w"].append(np.mean((chi - clo) / np.maximum(tc, 1e-12)))
                gd = [b.acc[:, i] - b.acc[:, j] for i, j in adj]
                r["gap_cov"].append(np.mean([
                    np.quantile(g, lo_q) <= ta[i] - ta[j] <= np.quantile(g, hi_q)
                    for g, (i, j) in zip(gd, adj)]))
                pd = b.dominance_prob()
                pf = b.frontier_prob()
                off = ~np.eye(len(ta), dtype=bool)
                cal[m].pred = np.append(cal[m].pred, pd[off])
                cal[m].actual = np.append(cal[m].actual, true_dom[off].astype(float))
                cert_dom = naive_dom & (pd >= thr)
                cert_front = naive_front & (pf >= thr)
            else:
                cert_dom, cert_front = naive_dom, naive_front
            dom_n[m] += int(cert_dom.sum())
            r["dom_true"].extend(true_dom[cert_dom].tolist())
            if cert_front.any():
                r["front_p"].append(true_front[cert_front].mean())
            r["front_r"].append(cert_front[true_front].mean() if true_front.any() else np.nan)

    rows = []
    for m in methods:
        r = acc_names[m]
        mean = lambda k: float(np.nanmean(r[k])) if r[k] else None  # noqa: E731
        rows.append({
            "scenario": scenario.name, "method": m,
            "acc_cov": mean("acc_cov"), "cost_cov": mean("cost_cov"),
            "acc_w": mean("acc_w"), "cost_w": mean("cost_w"), "gap_cov": mean("gap_cov"),
            "dom_prec": float(np.mean(r["dom_true"])) if r["dom_true"] else None,
            "dom_n": dom_n[m] / reps,
            "dom_ece": cal[m].ece() if m != "naive" else None,
            "front_prec": mean("front_p"), "front_recall": mean("front_r"),
            "reps": reps,
        })
    return rows


def format_rows(rows: list[dict]) -> str:
    def f(v, d=3):
        return "   --" if v is None else f"{v:.{d}f}"
    head = (f"{'method':14s}{'acc cov':>8s}{'width':>7s}{'cost cov':>9s}{'gap cov':>8s}"
            f"{'dom prec':>9s}{'dom/rep':>8s}{'dom ECE':>8s}{'front P':>8s}{'front R':>8s}")
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r['method']:14s}{f(r['acc_cov']):>8s}{f(r['acc_w']):>7s}"
                   f"{f(r['cost_cov']):>9s}{f(r['gap_cov']):>8s}{f(r['dom_prec']):>9s}"
                   f"{r['dom_n']:>8.1f}{f(r['dom_ece']):>8s}{f(r['front_prec']):>8s}"
                   f"{f(r['front_recall']):>8s}")
    return "\n".join(out)
