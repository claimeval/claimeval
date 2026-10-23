"""Tests for the baseline methods and the paper's comparison claims.

Run: python3 tests/test_compare.py   (also works under pytest)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claimeval import Rollout, RolloutSet, make_population, sample_dataset  # noqa: E402
from claimeval.baselines import METHODS, run_method  # noqa: E402

REAL = ROOT / "realdata"


def _rs(seed=0, n_tasks=60, clustered=False):
    from dataclasses import replace
    pop = make_population(6, np.random.default_rng(1))
    if clustered:
        pop = replace(pop, difficulty_dist="clustered", icc=0.6)
    return RolloutSet(sample_dataset(pop, n_tasks, 3, np.random.default_rng(seed)))


def test_every_method_returns_the_same_shape():
    rs = _rs()
    for m in METHODS:
        b = run_method(rs, m, B=50, rng=np.random.default_rng(0))
        assert b.acc.shape[1] == b.cost.shape[1] == rs.n_configs, m
        assert np.isfinite(b.acc).all() and np.isfinite(b.cost).all(), m


def test_formula_standard_errors_match_the_textbook():
    """wald's accuracy SE must equal sd(task means)/sqrt(T)."""
    rs = _rs(n_tasks=400)
    b = run_method(rs, "wald", B=20000, rng=np.random.default_rng(0), cost_stat="mean")
    acc_t = np.nanmean(rs.success, axis=2)
    se = acc_t.std(axis=1, ddof=1) / np.sqrt(acc_t.shape[1])
    assert np.allclose(b.acc.std(axis=0), se, rtol=0.05), (b.acc.std(axis=0), se)


def test_paired_and_bootstrap_agree_on_iid_data():
    """With independent tasks the paired normal approximation and the task
    bootstrap estimate the same variance: they must agree closely."""
    rs = _rs(n_tasks=300)
    p = run_method(rs, "paired_clt", B=4000, rng=np.random.default_rng(0), cost_stat="mean")
    t = run_method(rs, "boot_task", B=4000, rng=np.random.default_rng(0), cost_stat="mean")
    assert np.allclose(p.acc.std(0), t.acc.std(0), rtol=0.12)


def test_paired_methods_are_tighter_on_differences_than_wald():
    """Pairing removes the shared task difficulty from comparisons."""
    rs = _rs(n_tasks=200)
    w = run_method(rs, "wald", B=4000, rng=np.random.default_rng(0))
    p = run_method(rs, "paired_clt", B=4000, rng=np.random.default_rng(0))
    dw = (w.acc[:, 0] - w.acc[:, 1]).std()
    dp = (p.acc[:, 0] - p.acc[:, 1]).std()
    assert dp < dw, (dp, dw)


def test_clustered_formula_widens_on_clustered_data():
    rs = _rs(n_tasks=200, clustered=True)
    p = run_method(rs, "paired_clt", B=4000, rng=np.random.default_rng(0))
    c = run_method(rs, "clustered_clt", B=4000, rng=np.random.default_rng(0))
    assert c.acc.std(0).mean() > p.acc.std(0).mean()


def test_naive_has_no_spread():
    b = run_method(_rs(), "naive")
    assert b.acc.shape[0] == 1


def _load(name):
    p = REAL / name
    if not p.exists():
        print(f"    (skipped: {name} not generated)")
        return None
    return json.loads(p.read_text())


def _row(r, scen, m):
    return next(x for x in r["rows"] if x["scenario"] == scen and x["method"] == m)


def test_paper_comparison_claims_hold():
    """Each sentence of the comparison section, pinned to the results."""
    r = _load("compare_results.json")
    if r is None:
        return
    new, one, clu, std = ("SWE-bench-shaped, new-repos", "single run per task",
                          "clustered, ICC 0.9", "standard (50 tasks, 5 runs)")
    # ours better: new repositories, few unequal clusters
    assert _row(r, new, "boot_cluster_t")["acc_cov"] > _row(r, new, "clustered_clt")["acc_cov"] + 0.1
    assert _row(r, new, "boot_cluster_t")["gap_cov"] > _row(r, new, "clustered_clt")["gap_cov"] + 0.1
    # paper: on very unequal clusters ours is mildly BELOW nominal, two-stage nearer
    assert _row(r, new, "boot_cluster_t")["acc_cov"] < 0.95
    assert _row(r, new, "boot_group")["acc_cov"] > _row(r, new, "boot_cluster_t")["acc_cov"]
    # paper: with balanced clusters ours is the one near nominal on intervals AND gaps;
    # two-stage over-covers gaps, intact-unwidened under-covers
    for scen in ("clustered, ICC 0.6", "clustered, ICC 0.9"):
        assert abs(_row(r, scen, "boot_cluster_t")["acc_cov"] - 0.95) < 0.02, scen
        assert abs(_row(r, scen, "boot_cluster_t")["gap_cov"] - 0.95) < 0.02, scen
        assert _row(r, scen, "boot_group")["gap_cov"] > 0.97, scen
        assert _row(r, scen, "boot_cluster")["gap_cov"] < 0.92, scen
        assert _row(r, scen, "boot_cluster_t")["dom_n"] > _row(r, scen, "boot_group")["dom_n"], scen
        assert _row(r, scen, "boot_cluster_t")["dom_prec"] > _row(r, scen, "clustered_clt")["dom_prec"], scen
    # ours better: single run per task, dominance precision
    assert _row(r, one, "boot_task")["dom_prec"] > _row(r, one, "paired_clt")["dom_prec"] + 0.1
    # the existing method is better in power with balanced clusters (paper says so)
    assert _row(r, clu, "clustered_clt")["dom_n"] > _row(r, clu, "boot_cluster_t")["dom_n"]
    assert _row(r, clu, "clustered_clt")["front_recall"] > _row(r, clu, "boot_cluster_t")["front_recall"]
    # the paired approximation is at least as powerful in the standard setting
    assert _row(r, std, "paired_clt")["dom_n"] >= _row(r, std, "boot_task")["dom_n"]
    # every uncertainty method beats current practice on claim precision
    for scen in (new, one, clu, std):
        naive = _row(r, scen, "naive")["dom_prec"]
        for m in ("paired_clt", "clustered_clt", "boot_task", "boot_cluster_t"):
            assert _row(r, scen, m)["dom_prec"] >= naive, (scen, m)


def test_replication_claims_hold():
    r = _load("replication_results.json")
    if r is None:
        return
    res = r["results"]
    thr = str(r["threshold"])
    for key, v in res.items():
        naive = v["methods"]["naive"][thr]["replication"]
        for m in ("paired_clt", "clustered_clt", "boot_task", "boot_cluster_t"):
            x = v["methods"][m][thr]["replication"]
            if x is not None and v["methods"][m][thr]["claims_per_split"] >= 1:
                assert x >= naive - 0.02, (key, m, x, naive)


def test_replication_exceptions_are_as_the_paper_describes():
    """Paper: ours >= both normal approximations except on cost-frontier claims
    checked on held-out tasks. Any other exception means the text is wrong."""
    r = _load("replication_results.json")
    if r is None:
        return
    thr = str(r["threshold"])
    n_exc = 0
    for key, v in r["results"].items():
        name, claim, kind = key.split("|")
        ours = "boot_task" if kind == "task" else "boot_cluster_t"
        rr = {m: v["methods"][m][thr]["replication"] for m in (ours, "paired_clt", "clustered_clt")}
        if None in rr.values():
            continue
        gap = max(rr["paired_clt"], rr["clustered_clt"]) - rr[ours]
        if gap > 0:
            # paper: exceptions are cost-frontier claims on task splits (marginal),
            # plus at most ties to three decimals
            if gap >= 0.0005:
                assert claim == "dominance" and kind == "task", f"unexpected exception: {key} {rr}"
                assert gap < 0.03, (key, rr)
            n_exc += 1
    assert n_exc == 4, f"paper says four exceptions, data has {n_exc}"


def test_few_cluster_scale_matches_t_tables():
    from claimeval.bootstrap import few_cluster_scale, t_quantile
    assert abs(t_quantile(0.975, 9) - 2.262) < 0.002
    assert abs(t_quantile(0.975, 11) - 2.201) < 0.002
    assert abs(few_cluster_scale(10) - (10 / 9) ** 0.5 * 2.262 / 1.960) < 0.003
    assert few_cluster_scale(200) < 1.02  # many clusters: no widening


def test_intact_cluster_bootstrap_keeps_clusters_whole():
    from claimeval.bootstrap import _group_task_indices
    rs = _rs(n_tasks=60, clustered=True)
    idx, valid = _group_task_indices(rs, 20, np.random.default_rng(0), within=False)
    groups = [set(map(int, g)) for g in rs.group_tasks]
    for row, ok in zip(idx, valid if valid is not None else [None] * len(idx)):
        drawn = row[ok] if ok is not None else row
        pos = 0
        while pos < len(drawn):  # consecutive blocks must be whole clusters, in order
            g = next(g for g in groups if int(drawn[pos]) in g)
            block = list(map(int, drawn[pos:pos + len(g)]))
            assert sorted(block) == sorted(g), block
            pos += len(g)


def test_followups_hold():
    r = _load("extensions_results.json")
    if r is None:
        return
    rows = {x["runs"]: x for x in r["cost"]["rows"]}
    assert rows[1]["cov_run_estimand"] > 0.9 and rows[1]["cov_task_estimand"] < 0.5
    assert rows[20]["cov_task_estimand"] > 0.9 and rows[20]["cov_run_estimand"] < 0.1
    rk = r["ranking"]
    assert abs(rk["probability"]["5"] - rk["studentized"]["5"]) < 0.03  # paper: no difference


def test_error_bars_certify_fewer_than_paired_everywhere():
    """Paper: per-model error bars certify fewer claims than paired in every split."""
    r = _load("replication_results.json")
    if r is None:
        return
    thr = str(r["threshold"])
    for key, v in r["results"].items():
        w = v["methods"]["wald"][thr]["claims_per_split"]
        p = v["methods"]["paired_clt"][thr]["claims_per_split"]
        assert w < p, (key, w, p)


def test_power_claims_hold():
    """Paper, power at matched precision -- including where ours loses."""
    r = _load("power_results.json")
    if r is None:
        return
    pw = r["results"]
    at = lambda sc, m, t: pw[sc][m]["at_precision"][t]  # noqa: E731
    miller = lambda sc, t: max(at(sc, "paired_clt", t), at(sc, "clustered_clt", t))  # noqa: E731
    std, c6, c9, new = ("standard (50 tasks, 5 runs)", "clustered, ICC 0.6",
                        "clustered, ICC 0.9", "SWE-bench-shaped, new-repos")
    # no clusters: paired approximation more powerful at both precisions
    for t in ("0.95", "0.99"):
        assert at(std, "paired_clt", t) > at(std, "boot_task", t), t
    # clusters: tie at 95%, ours ahead at 99%
    for sc in (c6, c9):
        assert abs(at(sc, "boot_cluster_t", "0.95") - miller(sc, "0.95")) < 0.2, sc
        assert at(sc, "boot_cluster_t", "0.99") > 1.3 * miller(sc, "0.99"), sc
    # SWE-bench-shaped: tie at 95%, normal approximation ahead at 99%
    assert abs(at(new, "boot_cluster_t", "0.95") - miller(new, "0.95")) < 0.4
    assert miller(new, "0.99") > at(new, "boot_cluster_t", "0.99")


def test_calibrated_replication_claims_hold():
    r = _load("replication_calibration_results.json")
    if r is None:
        return
    res = r["results"]
    sf = lambda k, m: res[k][m]["predicted"] - res[k][m]["observed"]  # noqa: E731
    ml = {m: sf("SWE-bench multilingual|repo", m) - sf("SWE-bench multilingual|task", m)
          for m in ("paired_clt", "clustered_clt", "boot_cluster_t")}
    # paper: ours has the smallest excess on Multilingual, and it is "not zero"
    assert ml["boot_cluster_t"] < min(ml["paired_clt"], ml["clustered_clt"]), ml
    assert ml["boot_cluster_t"] > 0
    # paper: Multilingual low replication is "mostly" noise and selection
    total = sf("SWE-bench multilingual|repo", "boot_cluster_t")
    assert ml["boot_cluster_t"] < 0.5 * total, (ml, total)


def test_uneven_cluster_claims_hold():
    r = _load("uneven_results.json")
    if r is None:
        return
    row = lambda m, k: next(x for x in r["rows"]  # noqa: E731
                            if x["scenario"] == "SWE-bench-shaped, new-repos" and x["method"] == m)[k]
    assert row("boot_cluster_t", "acc_cov") < 0.95  # mildly below nominal
    assert row("boot_cluster_cr2", "acc_cov") < row("boot_cluster_t", "acc_cov")  # CR2 doesn't help
    assert row("boot_cluster_kish", "acc_cov") > 0.95 and row("boot_cluster_kish", "gap_cov") > 0.95
    assert row("boot_cluster_kish", "dom_n") < row("boot_cluster_t", "dom_n")
    for m in ("boot_cluster_t", "boot_cluster_cr2", "boot_cluster_kish"):  # all beat Miller
        assert row(m, "acc_cov") > row("clustered_clt", "acc_cov") + 0.05, m


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
