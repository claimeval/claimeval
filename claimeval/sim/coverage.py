"""Does the estimator earn the numbers it prints?

Every interval and probability in `claimeval` is a claim about long-run
frequency. This module checks those claims against a known population by
repeatedly drawing fresh task suites, so the paper can say "95% CI" and
"P(frontier) = 0.9" and mean them.

Four things are validated:
  1. marginal CI coverage for accuracy and cost;
  2. calibration of P(config on frontier);
  3. calibration of P(i dominates j);
  4. coverage of the Hausdorff frontier band;
plus the decision-level payoff: precision and recall of *certified* claims
against the naive point-estimate claims they are meant to replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from claimeval.stats.bootstrap import auto_mode, hierarchical_bootstrap
from claimeval.stats.frontier import dominance, frontier_band, frontier_points, hausdorff, log_cost_range, pareto_mask
from claimeval.core.schema import RolloutSet
from .simulate import Population, make_population, population_truth, sample_dataset

_BINS = np.linspace(0.0, 1.0, 11)


@dataclass
class Calibration:
    """Reliability of a probability forecast against binary outcomes."""

    pred: np.ndarray = field(default_factory=lambda: np.array([]))
    actual: np.ndarray = field(default_factory=lambda: np.array([]))

    def table(self) -> list[dict]:
        idx = np.clip(np.digitize(self.pred, _BINS) - 1, 0, len(_BINS) - 2)
        rows = []
        for b in range(len(_BINS) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append(
                {
                    "bin": f"[{_BINS[b]:.1f},{_BINS[b + 1]:.1f})",
                    "n": int(m.sum()),
                    "mean_pred": float(self.pred[m].mean()),
                    "empirical": float(self.actual[m].mean()),
                }
            )
        return rows

    def ece(self) -> float:
        """Expected calibration error, weighted by bin population."""
        rows = self.table()
        n = sum(r["n"] for r in rows)
        return sum(r["n"] * abs(r["mean_pred"] - r["empirical"]) for r in rows) / n if n else np.nan


def coverage_study(
    n_reps: int = 200,
    n_configs: int = 9,
    n_tasks: int = 50,
    n_seeds: int = 5,
    n_boot: int = 500,
    level: float = 0.95,
    band_level: float = 0.90,
    threshold: float = 0.90,
    band_candidates: int = 40,
    boot_mode: str | None = None,
    cost_stat: str = "median",
    pop: Population | None = None,
    seed: int = 0,
) -> dict:
    """Redraw the task suite `n_reps` times from a fixed population and score.

    The population (which configs exist, their ability and cost) is held fixed;
    tasks and rollouts are what get resampled. That matches the estimand the
    task-level bootstrap targets: performance on a task drawn from the suite's
    population, for this fixed set of configs.
    """
    rng = np.random.default_rng(seed)
    pop = pop or make_population(n_configs, np.random.default_rng(seed))
    true_acc, true_cost = population_truth(pop, cost_stat=cost_stat)
    true_mask = pareto_mask(true_cost, true_acc)
    true_dom = dominance(true_cost, true_acc)
    ref = log_cost_range(true_cost)
    true_front = frontier_points(true_cost, true_acc, ref)

    acc_hits, cost_hits, band_hits = [], [], []
    front_cal, dom_cal = Calibration(), Calibration()
    naive_p, naive_r, cert_p, cert_r = [], [], [], []
    naive_dom_p, cert_dom_p = [], []
    cert_empty = 0

    for _ in range(n_reps):
        rs = RolloutSet(sample_dataset(pop, n_tasks, n_seeds, rng))
        if rs.configs != pop.names():
            raise RuntimeError("config ordering drifted from population index")

        mode = boot_mode or auto_mode(rs)
        boot = hierarchical_bootstrap(rs, n_boot=n_boot, mode=mode, rng=rng,
                                      cost_stat=cost_stat)
        ci = boot.ci(level)
        acc_hits.append((ci["acc_lo"] <= true_acc) & (true_acc <= ci["acc_hi"]))
        cost_hits.append((ci["cost_lo"] <= true_cost) & (true_cost <= ci["cost_hi"]))

        p_front = boot.frontier_prob()
        front_cal.pred = np.append(front_cal.pred, p_front)
        front_cal.actual = np.append(front_cal.actual, true_mask.astype(float))

        p_dom = boot.dominance_prob()
        off = ~np.eye(rs.n_configs, dtype=bool)
        dom_cal.pred = np.append(dom_cal.pred, p_dom[off])
        dom_cal.actual = np.append(dom_cal.actual, true_dom[off].astype(float))

        # -- decision quality: naive claims vs certified claims ---------------
        acc_hat, cost_hat = rs.point_estimates(cost_stat)
        claimed = pareto_mask(cost_hat, acc_hat)
        certified = claimed & (p_front >= threshold)
        naive_p.append(_precision(claimed, true_mask))
        naive_r.append(_recall(claimed, true_mask))
        if certified.any():
            cert_p.append(_precision(certified, true_mask))
            cert_r.append(_recall(certified, true_mask))
        else:
            cert_empty += 1

        dom_claimed = dominance(cost_hat, acc_hat) & off
        naive_dom_p.append(_precision(dom_claimed, true_dom & off))
        dom_cert = dom_claimed & (p_dom >= threshold)
        if dom_cert.any():
            cert_dom_p.append(_precision(dom_cert, true_dom & off))

        band = frontier_band(
            boot.cost, boot.acc, ref, level=band_level, n_candidates=band_candidates, rng=rng
        )
        band_hits.append(hausdorff(true_front, band["median_front"]) <= band["radius"])

    return {
        "settings": {
            "n_reps": n_reps, "n_configs": pop.n_configs, "n_tasks": n_tasks,
            "n_seeds": n_seeds, "n_boot": n_boot, "level": level,
            "band_level": band_level, "threshold": threshold,
            "boot_mode": boot_mode or "auto", "cost_stat": cost_stat,
        },
        "configs": pop.names(),
        "true_frontier": true_mask,
        "acc_coverage": np.mean(acc_hits, axis=0),
        "cost_coverage": np.mean(cost_hits, axis=0),
        "acc_coverage_overall": float(np.mean(acc_hits)),
        "cost_coverage_overall": float(np.mean(cost_hits)),
        "band_coverage": float(np.mean(band_hits)),
        "frontier_calibration": front_cal,
        "dominance_calibration": dom_cal,
        "naive_precision": float(np.mean(naive_p)),
        "naive_recall": float(np.mean(naive_r)),
        "certified_precision": float(np.mean(cert_p)) if cert_p else np.nan,
        "certified_recall": float(np.mean(cert_r)) if cert_r else np.nan,
        "certified_empty_reps": cert_empty,
        "naive_dom_precision": float(np.mean(naive_dom_p)),
        "certified_dom_precision": float(np.mean(cert_dom_p)) if cert_dom_p else np.nan,
        "mc_se": float(np.sqrt(level * (1 - level) / n_reps)),
    }


FAMILIES: dict[str, dict] = {
    "baseline": {},
    "many configs": {"n_configs": 20},
    "few configs": {"n_configs": 5},
    "small suite": {"n_tasks": 20},
    "large suite": {"n_tasks": 200},
    "heavy cost tail": {"pop_kwargs": {"cost_sd": 0.75, "grind_frac": 0.85}},
    "heavy difficulty tail": {"pop_kwargs": {"difficulty_dist": "heavy", "heavy_df": 3.0}},
    "graded scores": {"pop_kwargs": {"partial_credit": 4.0}},
    "clustered (icc .6)": {"pop_kwargs": {"difficulty_dist": "clustered", "icc": 0.6}},
    "clustered (icc .9)": {"pop_kwargs": {"difficulty_dist": "clustered", "icc": 0.9}},
}


def robustness_sweep(
    families: dict[str, dict] | None = None,
    n_reps: int = 300,
    n_seeds: int = 5,
    n_boot: int = 600,
    modes: tuple[str, ...] = ("task", "cluster_t"),
    cost_stat: str = "median",
    seed: int = 0,
    **base,
) -> list[dict]:
    """Run the coverage study across generative families and bootstrap modes.

    Coverage validated on one population is an anecdote. This sweeps the
    structural assumptions most likely to be wrong in practice — how many
    configs, suite size, cost-tail weight, difficulty-tail weight, graded
    rather than binary scoring, and clustered task suites.
    """
    from dataclasses import replace

    families = families or FAMILIES
    rows = []
    for name, spec in families.items():
        spec = dict(spec)
        pop_kwargs = spec.pop("pop_kwargs", {})
        n_configs = spec.pop("n_configs", base.get("n_configs", 9))
        pop = make_population(n_configs, np.random.default_rng(seed))
        if pop_kwargs:
            pop = replace(pop, **pop_kwargs)
        for mode in modes:
            res = coverage_study(
                n_reps=n_reps, n_seeds=n_seeds, n_boot=n_boot,
                boot_mode=(None if mode == "auto" else mode), cost_stat=cost_stat,
                pop=pop, seed=seed, band_candidates=25, **{**base, **spec},
            )
            rows.append({
                "family": name, "mode": mode,
                "acc_cov": res["acc_coverage_overall"],
                "cost_cov": res["cost_coverage_overall"],
                "band_cov": res["band_coverage"],
                "frontier_ece": res["frontier_calibration"].ece(),
                "dom_ece": res["dominance_calibration"].ece(),
                "cert_precision": res["certified_precision"],
                "naive_precision": res["naive_precision"],
                "mc_se": res["mc_se"],
            })
    return rows


def format_sweep(rows: list[dict], level: float = 0.95) -> str:
    modes = sorted({r["mode"] for r in rows})
    se = rows[0]["mc_se"] if rows else 0.0
    L = [
        "=" * 88,
        "ROBUSTNESS SWEEP",
        "=" * 88,
        f"nominal CI coverage {level:.2f}; MC SE +/-{se:.3f}. "
        "Flagged when |coverage - nominal| > 2 SE.",
        "",
        f"{'family':<22}{'mode':<7}{'acc cov':>9}{'cost cov':>10}{'band':>8}"
        f"{'front ECE':>11}{'dom ECE':>9}{'cert prec':>11}",
        "-" * 88,
    ]
    for r in rows:
        flag = "  <--" if abs(r["acc_cov"] - level) > 2 * r["mc_se"] else ""
        L.append(
            f"{r['family'][:22]:<22}{r['mode']:<7}{r['acc_cov']:>9.3f}{r['cost_cov']:>10.3f}"
            f"{r['band_cov']:>8.3f}{r['frontier_ece']:>11.4f}{r['dom_ece']:>9.4f}"
            f"{r['cert_precision']:>11.3f}{flag}"
        )
    L += ["", f"modes compared: {', '.join(modes)}", "=" * 88]
    return "\n".join(L)


def _precision(claimed: np.ndarray, truth: np.ndarray) -> float:
    return float(truth[claimed].mean()) if claimed.any() else np.nan


def _recall(claimed: np.ndarray, truth: np.ndarray) -> float:
    return float(claimed[truth].mean()) if truth.any() else np.nan


def format_coverage(res: dict) -> str:
    s = res["settings"]
    L = [
        "=" * 78,
        "COVERAGE STUDY",
        "=" * 78,
        f"{s['n_reps']} replications | {s['n_configs']} configs x {s['n_tasks']} tasks "
        f"x {s['n_seeds']} seeds | {s['n_boot']} bootstrap draws",
        f"Monte Carlo SE on a {s['level']:.0%} coverage estimate: "
        f"+/-{res['mc_se']:.3f} (so read +/-{2 * res['mc_se']:.3f})",
        "",
        "-" * 78,
        f"1. MARGINAL CI COVERAGE  (nominal {s['level']:.0%})",
        "-" * 78,
        f"{'config':<26}{'accuracy':>12}{'cost':>12}{'on true frontier':>20}",
    ]
    for i, c in enumerate(res["configs"]):
        L.append(
            f"{c[:26]:<26}{res['acc_coverage'][i]:>12.3f}{res['cost_coverage'][i]:>12.3f}"
            f"{('yes' if res['true_frontier'][i] else '-'):>20}"
        )
    L += [
        "",
        f"overall: accuracy {res['acc_coverage_overall']:.3f}, "
        f"cost {res['cost_coverage_overall']:.3f}  (nominal {s['level']:.2f})",
    ]

    for title, key in (
        ("2. CALIBRATION OF P(config on frontier)", "frontier_calibration"),
        ("3. CALIBRATION OF P(i dominates j)", "dominance_calibration"),
    ):
        cal = res[key]
        L += ["", "-" * 78, title, "-" * 78,
              f"{'predicted bin':<18}{'n':>8}{'mean pred':>12}{'empirical':>12}"]
        for r in cal.table():
            L.append(f"{r['bin']:<18}{r['n']:>8}{r['mean_pred']:>12.3f}{r['empirical']:>12.3f}")
        L.append(f"ECE = {cal.ece():.4f}")

    L += [
        "",
        "-" * 78,
        f"4. FRONTIER BAND COVERAGE  (nominal {s['band_level']:.0%})",
        "-" * 78,
        f"true frontier inside the band in {res['band_coverage']:.3f} of replications",
        "",
        "-" * 78,
        f"5. DECISION QUALITY  (certification threshold P>={s['threshold']:.2f})",
        "-" * 78,
        f"{'claim type':<34}{'precision':>12}{'recall':>12}",
        f"{'naive frontier claims':<34}{res['naive_precision']:>12.3f}{res['naive_recall']:>12.3f}",
        f"{'certified frontier claims':<34}{res['certified_precision']:>12.3f}"
        f"{res['certified_recall']:>12.3f}",
        f"{'naive dominance claims':<34}{res['naive_dom_precision']:>12.3f}{'':>12}",
        f"{'certified dominance claims':<34}{res['certified_dom_precision']:>12.3f}{'':>12}",
    ]
    if res["certified_empty_reps"]:
        L.append(
            f"note: {res['certified_empty_reps']}/{s['n_reps']} replications certified "
            "nothing (excluded from certified precision)"
        )
    L += [
        "",
        "Precision = share of claims that are true. Recall = share of true frontier",
        "members claimed. Certification should trade recall for precision; if it does",
        "not, the probabilities are not doing their job.",
        "=" * 78,
    ]
    return "\n".join(L)
