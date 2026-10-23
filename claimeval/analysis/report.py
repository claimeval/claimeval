"""The E1 headline: how many published Pareto-optimal claims survive an error bar."""

from __future__ import annotations

import numpy as np

from claimeval.stats.bootstrap import BootstrapResult, variance_decomposition
from claimeval.stats.cost_axis import diagnose_cost_axis, format_diagnosis
from claimeval.stats.frontier import dominance, frontier_band, pareto_mask
from claimeval.core.schema import RolloutSet


def headline(rs: RolloutSet, boot: BootstrapResult, threshold: float = 0.90) -> dict:
    """Audit the frontier claims a naive analysis of this data would make.

    Two claim types are audited separately because papers make both:
      - membership: "config X is Pareto-optimal"
      - dominance:  "config X dominates config Y"
    A claim is *supported* if its bootstrap probability clears `threshold`.
    """
    acc, cost = rs.point_estimates()
    observed = pareto_mask(cost, acc)
    p_front = boot.frontier_prob()

    obs_idx = np.flatnonzero(observed)
    supported = p_front[obs_idx] >= threshold

    obs_dom = dominance(cost, acc)  # (C, C) claims the point estimates make
    p_dom = boot.dominance_prob()
    dom_pairs = np.argwhere(obs_dom)
    dom_supported = (
        np.array([p_dom[i, j] >= threshold for i, j in dom_pairs])
        if len(dom_pairs)
        else np.array([], dtype=bool)
    )

    return {
        "threshold": threshold,
        "n_configs": rs.n_configs,
        "frontier_claims": len(obs_idx),
        "frontier_supported": int(supported.sum()),
        "frontier_spurious_frac": (
            float(1 - supported.mean()) if len(obs_idx) else float("nan")
        ),
        "dominance_claims": len(dom_pairs),
        "dominance_supported": int(dom_supported.sum()),
        "dominance_spurious_frac": (
            float(1 - dom_supported.mean()) if len(dom_pairs) else float("nan")
        ),
        "per_config": [
            {
                "config": rs.configs[i],
                "acc": float(acc[i]),
                "cost": float(cost[i]),
                "p_frontier": float(p_front[i]),
                "observed_frontier": bool(observed[i]),
            }
            for i in range(rs.n_configs)
        ],
    }


def format_report(
    rs: RolloutSet,
    boot: BootstrapResult,
    head: dict,
    bias: dict | None = None,
    decomp: dict | None = None,
    band: dict | None = None,
) -> str:
    ci = boot.ci(0.95)
    p_front = boot.frontier_prob()
    acc, cost = rs.point_estimates()
    cost_mean = rs.point_estimates('mean')[1]
    cps = rs.cost_per_success()
    order = np.argsort(cost)

    L = ["=" * 78, "PARETO FRONTIER AUDIT", "=" * 78, "", rs.describe(), ""]
    L += [
        f"{boot.n_replicates} bootstrap replicates (mode={boot.mode})",
        "",
        f"{'config':<26}{'acc [95% CI]':>22}{'cost~':>10}{'costbar':>10}"
        f"{'$/succ':>9}{'P(front)':>10}{'':>4}",
        "-" * 78,
    ]
    for i in order:
        flag = "*" if head["per_config"][i]["observed_frontier"] else " "
        warn = "  <-- unsupported" if (flag == "*" and p_front[i] < head["threshold"]) else ""
        L.append(
            f"{rs.configs[i][:26]:<26}"
            f"{acc[i]:>8.3f} [{ci['acc_lo'][i]:.3f},{ci['acc_hi'][i]:.3f}]"
            f"{cost[i]:>10.4f}{cost_mean[i]:>10.4f}{cps[i]:>9.3f}{p_front[i]:>10.2f}  {flag}{warn}"
        )
    L += ["", "cost~ = median per-task cost (the frontier axis); costbar = mean",
          "* = on the frontier of the point estimates (what a paper would report)", "",
          "-" * 78, "COST AXIS DIAGNOSIS", "-" * 78,
          format_diagnosis(diagnose_cost_axis(rs)), ""]

    L += ["-" * 78, "HEADLINE", "-" * 78]
    L.append(
        f"Pareto-optimality claims:  {head['frontier_supported']}/"
        f"{head['frontier_claims']} supported at P>={head['threshold']:.2f}"
        f"   ({head['frontier_spurious_frac']:.0%} unsupported)"
    )
    L.append(
        f"Pairwise dominance claims: {head['dominance_supported']}/"
        f"{head['dominance_claims']} supported"
        f"   ({head['dominance_spurious_frac']:.0%} unsupported)"
    )

    if bias is not None:
        L += ["", "-" * 78, "SELECTION BIAS (winner's curse)", "-" * 78]
        if not bias.get("available"):
            L.append(f"not estimable: {bias['reason']}")
        else:
            L.append(
                f"accuracy inflation on selected configs: "
                f"{bias['acc_bias']:+.4f} (SE {bias['acc_bias_se']:.4f})"
                f"  = {bias['rel_acc_bias']:+.1%} relative"
            )
            L.append(f"cost understatement:                    {bias['cost_bias']:+.4f}")
            L.append(
                f"mean frontier size {bias['mean_frontier_size']:.1f} "
                f"over {bias['n_splits']} splits"
            )

    if decomp is not None:
        L += ["", "-" * 78, "VARIANCE DECOMPOSITION", "-" * 78]
        L.append(f"{'config':<26}{'sd_total':>10}{'sd_task':>10}{'sd_seed':>10}{'task share':>12}")
        for i in order:
            L.append(
                f"{rs.configs[i][:26]:<26}{decomp['sd_total'][i]:>10.4f}"
                f"{decomp['sd_task'][i]:>10.4f}{decomp['sd_seed'][i]:>10.4f}"
                f"{decomp['task_share'][i]:>12.2f}"
            )
        L.append("")
        L.append("task share -> 1: buy more tasks.  -> 0: buy more seeds.")

    if band is not None:
        L += [
            "",
            "-" * 78,
            "FRONTIER BAND",
            "-" * 78,
            f"{band['level']:.0%} Hausdorff band radius: {band['radius']:.4f} "
            f"(normalized log-cost x accuracy space)",
            f"median front has {len(band['median_front'])} points",
        ]

    L.append("=" * 78)
    return "\n".join(L)


def analyze(
    rs: RolloutSet,
    n_boot: int = 2000,
    threshold: float = 0.90,
    with_bias: bool = True,
    with_decomp: bool = True,
    with_band: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[str, dict]:
    """Full E1 pipeline on one dataset."""
    from claimeval.stats.bias import split_sample_debias
    from claimeval.stats.bootstrap import auto_mode, hierarchical_bootstrap

    rng = rng or np.random.default_rng(0)
    # auto_mode: "group" when the suite is clustered, else "task"; never
    # "both", which double-counts the within-cell component
    boot = hierarchical_bootstrap(rs, n_boot=n_boot, mode=auto_mode(rs), rng=rng)
    head = headline(rs, boot, threshold)
    bias = split_sample_debias(rs, rng=rng) if with_bias else None
    decomp = variance_decomposition(rs, n_boot=min(n_boot, 1000), rng=rng) if with_decomp else None
    band = (
        frontier_band(boot.cost, boot.acc, boot.log_cost_ref(), rng=rng)
        if with_band
        else None
    )
    return format_report(rs, boot, head, bias, decomp, band), head
