"""Run: python3 tests/test_claimeval.py   (also works under pytest)"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimeval import (  # noqa: E402
    Rollout,
    RolloutSet,
    hausdorff,
    hierarchical_bootstrap,
    pareto_mask,
    seed_count_curve,
    simulate,
    split_sample_debias,
    variance_decomposition,
)
from claimeval.frontier import dominance  # noqa: E402
from claimeval.ingest import guess_map, inspect_path, load  # noqa: E402
from claimeval.report import headline  # noqa: E402


def test_pareto_mask_basic():
    cost = np.array([1.0, 2.0, 3.0, 2.0])
    acc = np.array([0.5, 0.7, 0.6, 0.4])
    # idx3 (cost 2, acc .4) is dominated by idx1 (cost 2, acc .7)
    # idx2 (cost 3, acc .6) is dominated by idx1 (cost 2, acc .7)
    assert pareto_mask(cost, acc).tolist() == [True, True, False, False]


def test_exact_ties_both_retained():
    cost = np.array([1.0, 1.0])
    acc = np.array([0.5, 0.5])
    assert pareto_mask(cost, acc).tolist() == [True, True]
    assert not dominance(cost, acc).any()


def test_dominance_is_batched():
    cost = np.random.default_rng(0).random((7, 5)) + 0.1
    acc = np.random.default_rng(1).random((7, 5))
    assert dominance(cost, acc).shape == (7, 5, 5)
    assert pareto_mask(cost, acc).shape == (7, 5)


def test_hausdorff_symmetry_and_zero():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    assert hausdorff(a, a) == 0.0
    assert abs(hausdorff(a, b) - hausdorff(b, a)) < 1e-12


def test_ragged_cells_are_kept():
    rolls = [
        Rollout("a", "t1", "0", 1.0, 1.0),
        Rollout("a", "t1", "1", 0.0, 5.0),
        Rollout("a", "t2", "0", 1.0, 2.0),
        Rollout("b", "t1", "0", 0.0, 0.5),  # b never ran t2
    ]
    rs = RolloutSet(rolls)
    assert rs.counts.tolist() == [[2, 1], [1, 0]]
    acc, cost = rs.point_estimates()
    assert abs(acc[0] - 0.75) < 1e-12  # mean of task means: (0.5, 1.0)
    assert rs.coverage().tolist() == [1.0, 0.5]


def test_cost_per_success_infinite_when_never_solves():
    rs = RolloutSet([Rollout("a", "t1", "0", 0.0, 3.0)])
    assert np.isinf(rs.cost_per_success()[0])


def test_seed_only_variance_is_smaller_than_total():
    rolls, _ = simulate(6, 30, 8, np.random.default_rng(3))
    rs = RolloutSet(rolls)
    rng = np.random.default_rng(0)
    both = hierarchical_bootstrap(rs, 400, "both", rng)
    seed = hierarchical_bootstrap(rs, 400, "seed", rng)
    assert (seed.acc.std(axis=0) <= both.acc.std(axis=0) + 1e-9).all()


def test_variance_decomposition_shares_in_range():
    rolls, _ = simulate(5, 25, 6, np.random.default_rng(4))
    d = variance_decomposition(RolloutSet(rolls), n_boot=300)
    assert np.all((d["task_share"] >= 0) & (d["task_share"] <= 1))


def test_frontier_prob_is_a_probability_and_ranks_sensibly():
    rolls, _ = simulate(8, 40, 10, np.random.default_rng(5))
    rs = RolloutSet(rolls)
    boot = hierarchical_bootstrap(rs, 600, "both", np.random.default_rng(0))
    p = boot.frontier_prob()
    assert np.all((p >= 0) & (p <= 1))
    acc, cost = rs.point_estimates()
    on = pareto_mask(cost, acc)
    assert p[on].mean() > p[~on].mean()  # observed members should score higher


def test_headline_counts_are_consistent():
    rolls, _ = simulate(8, 40, 10, np.random.default_rng(6))
    rs = RolloutSet(rolls)
    boot = hierarchical_bootstrap(rs, 600, "both", np.random.default_rng(0))
    h = headline(rs, boot, 0.9)
    assert 0 <= h["frontier_supported"] <= h["frontier_claims"] <= h["n_configs"]
    assert 0.0 <= h["frontier_spurious_frac"] <= 1.0


def test_winners_curse_is_positive():
    """Selecting on noise inflates the selected configs' accuracy."""
    rolls, _ = simulate(9, 50, 12, np.random.default_rng(7))
    b = split_sample_debias(RolloutSet(rolls), n_splits=120, rng=np.random.default_rng(0))
    assert b["available"]
    assert b["acc_bias"] > 0, b


def test_debias_unavailable_with_single_seed():
    rs = RolloutSet([Rollout("a", "t1", "0", 1.0, 1.0), Rollout("b", "t1", "0", 0.0, 2.0)])
    assert split_sample_debias(rs)["available"] is False


def test_seed_curve_improves_with_k():
    rolls, _ = simulate(8, 40, 16, np.random.default_rng(8))
    rows = seed_count_curve(RolloutSet(rolls), (1, 4, 16), n_repeats=25, rng=np.random.default_rng(0))
    assert rows[0]["jaccard_vs_full"] < rows[-1]["jaccard_vs_full"]
    assert rows[0]["acc_inflation"] > rows[-1]["acc_inflation"]
    assert rows[0]["hausdorff_vs_full"] > rows[-1]["hausdorff_vs_full"]


def test_ingest_roundtrip_and_guessing():
    recs = [
        {"agent_name": "swe-agent", "model_name": "m1", "instance_id": f"t{i}",
         "resolved": i % 2 == 0, "total_cost": 0.5 + 0.1 * i, "run_id": "r0"}
        for i in range(6)
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "results.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs))
        info = inspect_path(p)
        fmap = info["guess"]
        assert fmap.config == ["agent_name", "model_name"]
        assert fmap.task == "instance_id" and fmap.success == "resolved"
        assert fmap.cost == "total_cost"
        rs, dropped = load(p, fmap)
        assert not dropped and rs.n_tasks == 6 and rs.n_configs == 1
        acc, _ = rs.point_estimates()
        assert abs(acc[0] - 0.5) < 1e-12


def test_ingest_task_keyed_dict_shape():
    """The documented HAL agent-output shape: {task_id: {...}}."""
    blob = {f"t{i}": {"cost": 0.2 * (i + 1), "score": float(i > 1)} for i in range(4)}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.json"
        p.write_text(json.dumps(blob))
        info = inspect_path(p)
        assert info["n_records"] == 4
        fmap = info["guess"]
        assert fmap.task == "_key"
        rs, _ = load(p, fmap)
        assert rs.n_tasks == 4


def test_ingest_counts_dropped_records():
    recs = [{"task_id": "t1", "score": 1, "cost": 1.0}, {"task_id": "t2", "score": 1}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs))
        fmap = guess_map(inspect_path(p)["keys"])
        _, dropped = load(p, fmap)
        assert dropped == {"no_cost": 1}


def test_population_truth_matches_sampler():
    """The closed form must not drift from the code path the estimator sees."""
    from claimeval import make_population, population_truth, sample_dataset

    pop = make_population(5, np.random.default_rng(11))
    acc_t, cost_t = population_truth(pop, n_mc=200_000, cost_stat="mean")
    rs = RolloutSet(sample_dataset(pop, 3000, 6, np.random.default_rng(12)))
    acc_e, cost_e = rs.point_estimates("mean")
    assert np.max(np.abs(acc_t - acc_e)) < 0.02, (acc_t, acc_e)
    assert np.max(np.abs(cost_t - cost_e) / cost_t) < 0.03, (cost_t, cost_e)


def test_median_cost_has_small_seed_bias_that_shrinks():
    """The price of a median cost axis: biased at low seed counts, unlike the mean.

    Mean cost is unbiased but its estimand can fail to exist under heavy tails
    (see test_mean_cost_estimand_diverges_under_heavy_tails). Median always
    exists but the median of noisy per-task means is not the median of true
    per-task means until seeds accumulate.
    """
    from claimeval import make_population, population_truth, sample_dataset

    pop = make_population(5, np.random.default_rng(11))
    truth = population_truth(pop, n_mc=200_000, cost_stat="median")[1]
    errs = []
    for ns in (6, 24):
        rs = RolloutSet(sample_dataset(pop, 3000, ns, np.random.default_rng(12)))
        errs.append(np.max(np.abs(truth - rs.point_estimates("median")[1]) / truth))
    assert errs[0] < 0.08, errs
    assert errs[1] < 0.02, errs
    assert errs[1] < errs[0], errs


def test_mean_cost_estimand_diverges_under_heavy_tails():
    """Why median is the default: E[10^(k*D)] is infinite for polynomial-tailed D.

    The Student-t has no MGF, so the population mean cost does not exist. A
    finite sample reports a stable-looking finite number regardless, which is
    precisely the trap.
    """
    from dataclasses import replace

    from claimeval import make_population, population_truth

    pop = replace(make_population(5, np.random.default_rng(0)),
                  difficulty_dist="heavy", heavy_df=3.0)
    means = [population_truth(pop, n_mc=100_000, cost_stat="mean",
                              rng=np.random.default_rng(s))[1][2] for s in range(6)]
    medians = [population_truth(pop, n_mc=100_000, cost_stat="median",
                                rng=np.random.default_rng(s))[1][2] for s in range(6)]
    assert max(means) / min(means) > 50, means  # no convergence
    assert max(medians) / min(medians) < 1.1, medians  # stable


def test_coverage_study_smoke():
    from claimeval import coverage_study, format_coverage

    res = coverage_study(n_reps=8, n_configs=4, n_tasks=15, n_seeds=3, n_boot=120,
                         band_candidates=15)
    assert 0.0 <= res["acc_coverage_overall"] <= 1.0
    assert 0.0 <= res["band_coverage"] <= 1.0
    assert res["frontier_calibration"].pred.size == 8 * 4
    assert 0.0 <= res["frontier_calibration"].ece() <= 1.0
    assert isinstance(format_coverage(res), str)


def test_task_bootstrap_attains_nominal_coverage():
    """Regression guard on the headline methodological finding.

    Task-only resampling should sit near nominal; resampling seeds as well
    double-counts the within-cell component and over-covers.
    """
    from claimeval import coverage_study

    task = coverage_study(n_reps=150, n_seeds=4, n_boot=500, boot_mode="task",
                          band_candidates=20)["acc_coverage_overall"]
    both = coverage_study(n_reps=150, n_seeds=4, n_boot=500, boot_mode="both",
                          band_candidates=20)["acc_coverage_overall"]
    assert 0.91 <= task <= 0.98, task
    assert both > task, (both, task)


def test_variance_decomposition_does_not_double_count():
    """sd_total must be the task-only SD, not the sum of the two components."""
    rolls, _ = simulate(5, 30, 6, np.random.default_rng(21))
    rs = RolloutSet(rolls)
    d = variance_decomposition(rs, n_boot=500)
    total = d["sd_total"] ** 2
    parts = d["sd_task"] ** 2 + d["sd_seed"] ** 2
    assert np.allclose(total, parts, atol=1e-9), (total, parts)


def test_group_bootstrap_fixes_clustered_undercoverage():
    """Task-level resampling fails on clustered suites; cluster resampling fixes it."""
    from dataclasses import replace

    from claimeval import coverage_study, make_population

    pop = replace(make_population(9, np.random.default_rng(0)),
                  difficulty_dist="clustered", icc=0.9)
    task = coverage_study(n_reps=120, n_seeds=4, n_boot=400, pop=pop,
                          boot_mode="task", band_candidates=15)["acc_coverage_overall"]
    grp = coverage_study(n_reps=120, n_seeds=4, n_boot=400, pop=pop,
                         boot_mode="group", band_candidates=15)["acc_coverage_overall"]
    assert task < 0.85, task  # badly under-covers
    assert grp > 0.89, grp  # restored
    assert grp - task > 0.10, (grp, task)


def test_auto_mode_picks_group_only_when_clustered():
    from claimeval import Rollout, RolloutSet
    from claimeval.bootstrap import auto_mode

    flat = RolloutSet([Rollout("a", f"t{i}", "0", 1.0, 1.0) for i in range(4)])
    assert not flat.is_clustered and auto_mode(flat) == "task"
    clustered = RolloutSet(
        [Rollout("a", f"t{i}", "0", 1.0, 1.0, group=f"g{i % 2}") for i in range(4)]
    )
    assert clustered.is_clustered and auto_mode(clustered) == "cluster_t"
    assert clustered.n_groups == 2


def test_cost_axis_diagnostic_flags_median_bias_regime():
    """The detectable pathology must be detected; thresholds are calibrated on it."""
    from dataclasses import replace

    from claimeval import make_population, sample_dataset
    from claimeval.cost_axis import diagnose_cost_axis

    base = make_population(9, np.random.default_rng(0))
    heavy = replace(base, cost_sd=0.75, grind_frac=0.85)
    d_ok = diagnose_cost_axis(RolloutSet(sample_dataset(base, 60, 20, np.random.default_rng(3))))
    d_bad = diagnose_cost_axis(RolloutSet(sample_dataset(heavy, 60, 5, np.random.default_rng(3))))
    assert d_ok["recommend"] == "either", d_ok
    assert d_bad["recommend"] == "mean", d_bad
    assert d_bad["median_bias_risk"] > 2 * d_ok["median_bias_risk"]


def test_calibration_is_perfect_on_a_perfect_forecaster():
    from claimeval.coverage import Calibration

    rng = np.random.default_rng(0)
    p = rng.random(20000)
    cal = Calibration(pred=p, actual=(rng.random(20000) < p).astype(float))
    assert cal.ece() < 0.02


def _groups(sizes):
    out, p = [], 0
    for s in sizes:
        out.append(np.arange(p, p + s))
        p += s
    return out


def test_wild_weights_enumerate_exactly():
    from claimeval.wild import _weights

    rng = np.random.default_rng(0)
    w = _weights(12, rng)
    assert w.shape == (4096, 12) and len({tuple(r) for r in w}) == 4096
    assert set(np.unique(w)) == {-1.0, 1.0}
    w6 = _weights(6, rng)  # Webb six-point weights below 10 clusters
    assert w6.shape == (6 ** 6, 6) and abs(w6.mean()) < 1e-12 and abs((w6 ** 2).mean() - 1) < 1e-12


def test_wcr_contains_estimate_and_matches_cr1_with_many_equal_clusters():
    from claimeval.wild import cr1_t_interval, wcr_interval

    rng = np.random.default_rng(1)
    g = _groups([20] * 14)
    z = (rng.random(280) < 0.4).astype(float)
    lo, hi = wcr_interval(z, g, rng)
    assert lo < z.mean() < hi
    lo2, hi2 = cr1_t_interval(z, g)
    assert abs((hi - lo) / (hi2 - lo2) - 1) < 0.2


def test_wcr_holds_coverage_where_cr1_fails_with_a_dominant_cluster():
    from claimeval.wild import cr1_t_interval, wcr_interval

    rng = np.random.default_rng(2)
    sizes = [350] + [14] * 11
    g = _groups(sizes)
    hits_w, hits_c = [], []
    for _ in range(150):
        eff = np.repeat(rng.normal(0, 1.0, len(sizes)), sizes)
        z = eff + rng.normal(0, 1.0, sum(sizes))  # truth: mean 0
        lo, hi = wcr_interval(z, g, rng)
        hits_w.append(lo <= 0 <= hi)
        lo, hi = cr1_t_interval(z, g)
        hits_c.append(lo <= 0 <= hi)
    assert np.mean(hits_w) > 0.9 and np.mean(hits_c) < 0.8


def test_isotonic_is_monotone_and_reproduces_sorted_data():
    from claimeval.calibrate import isotonic

    rng = np.random.default_rng(3)
    x = rng.random(500)
    y = (rng.random(500) < x).astype(float)
    k, v = isotonic(x, y)
    assert np.all(np.diff(v) >= -1e-12) and np.all(np.diff(k) >= 0)
    k2, v2 = isotonic(np.arange(5.0), np.arange(5.0))
    assert np.allclose(v2, np.arange(5.0))


def test_subset_keeps_cells_and_relabels_repeats():
    from claimeval.calibrate import subset

    rs = RolloutSet(simulate(n_configs=3, n_tasks=10, n_seeds=2, rng=np.random.default_rng(0))[0])
    idx = np.array([0, 0, 3])
    s = rs.success.copy()
    sub = subset(rs, idx)
    assert sub.n_tasks == 3 and np.array_equal(sub.success[:, 1], s[:, 0], equal_nan=True)
    a, _ = sub.point_estimates()
    assert np.allclose(a, np.nanmean(np.nanmean(rs.success[:, idx], axis=2), axis=1))


def _two_agents(gap, n=400, seed=0):
    rng = np.random.default_rng(seed)
    rolls = []
    for t in range(n):
        d = rng.normal()
        for name, skill, cost in (("strong", gap, 2.0), ("weak", 0.0, 1.0), ("cheap", gap, 0.5)):
            p = 1 / (1 + np.exp(-(skill - d)))
            rolls.append(Rollout(name, f"t{t}", "0", float(rng.random() < p), cost * np.exp(rng.normal(0, .1))))
    return RolloutSet(rolls)


def test_claims_large_gap_supported_and_tie_not():
    from claimeval.claims import Claim, evaluate_claims

    big = evaluate_claims(_two_agents(1.5), [Claim("pairwise", "strong", ["weak"])], n_boot=1000)[0]
    assert big["supported"] and big["probability"] > 0.99
    tie = evaluate_claims(_two_agents(1.5), [Claim("pairwise", "strong", ["cheap"])], n_boot=1000)[0]
    assert not tie["supported"] and 0.05 < tie["probability"] < 0.95


def test_claims_kinds_consistent():
    from claimeval.claims import Claim, evaluate_claims

    rs = _two_agents(1.5)
    r = {x["id"]: x for x in evaluate_claims(rs, [
        Claim("cost_dominance", "cheap", ["strong"], id="dom"),
        Claim("frontier", "weak", ["strong", "cheap"], id="front"),
        Claim("margin", "strong", ["weak"], margin=0.9, id="margin"),
        Claim("top1", "weak", ["strong", "cheap"], id="top")], n_boot=1000)}
    assert r["front"]["probability"] < 0.1 and r["top"]["probability"] == 0.0
    assert r["margin"]["probability"] == 0.0, "a 90-point margin cannot hold"
    try:
        Claim("best", "a")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind accepted")


def test_adapters_read_public_formats():
    import json
    import tempfile
    from claimeval import adapters

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # SWE-bench experiments layout: <sub>/results/results.json
        for sub, solved in (("a", ["r__x-1", "r__x-2"]), ("b", ["q__y-3"])):
            (td / sub / "results").mkdir(parents=True)
            (td / sub / "results" / "results.json").write_text(json.dumps(
                {"resolved": solved, "no_generation": ["q__y-3"] if sub == "a" else ["r__x-1"]}))
        (td / "c" / "results").mkdir(parents=True)
        (td / "c" / "results" / "results.json").write_text(json.dumps({"resolved": 7}))
        rs, sk = adapters.swebench(td)
        assert rs.n_tasks == 3 and rs.n_groups == 2 and "c" in sk
        acc, _ = rs.point_estimates()
        assert np.allclose(acc, [2 / 3, 1 / 3])
        # tau2 results file
        f = td / "m_airline.json"
        f.write_text(json.dumps({"simulations": [
            {"task_id": t, "trial": k, "reward_info": {"reward": float(t == "1")}, "agent_cost": 0.1}
            for t in ("1", "2") for k in range(3)]}))
        rs, sk = adapters.tau2([f])
        assert rs.max_seeds == 3 and rs.point_estimates()[0][0] == 0.5 and not sk
        # Harbor (Terminal-Bench) layout: <sub>/<job>/<trial>/result.json
        for i, rew in enumerate((1.0, None)):
            p = td / "tb" / "agentX" / "job1" / f"t{i}"
            p.mkdir(parents=True)
            (p / "result.json").write_text(json.dumps({
                "task_name": f"terminal-bench/task{i}", "task_id": {"git_commit_id": "abc"},
                "verifier_result": {"rewards": {"reward": rew}}, "agent_result": {"cost_usd": 0.5}}))
        (td / "tb" / "agentX" / "job1" / "result.json").write_text("{}")
        rs, _ = adapters.terminal_bench(td / "tb")
        assert rs.n_tasks == 2 and rs.point_estimates()[0][0] == 0.5, "unscored trial counts as 0"


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
