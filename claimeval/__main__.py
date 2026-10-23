"""CLI: claimeval {demo,inspect,analyze,claims,seed-curve,coverage}"""

from __future__ import annotations

import argparse
import glob
import os
import json
import sys

import numpy as np

from claimeval.stats.bias import seed_count_curve
from claimeval.core.ingest import FieldMap, guess_map, inspect_path, load
from claimeval.analysis.report import analyze
from claimeval.core.schema import RolloutSet
from claimeval.sim.simulate import simulate


def _load(args) -> RolloutSet:
    src = getattr(args, "source", None)
    if src:
        from claimeval.core import adapters
        if src == "tau2":
            rs, skipped = adapters.tau2(sorted(glob.glob(os.path.join(args.path, "*.json")))
                                        if os.path.isdir(args.path) else [args.path])
        elif src == "swebench-cost":
            rs, skipped = adapters.swebench(args.path, with_cost=True)
        else:
            rs, skipped = {"swebench": adapters.swebench,
                           "terminal-bench": adapters.terminal_bench}[src](args.path)
        for k, v in skipped.items():
            print(f"[skipped] {k}: {v}", file=sys.stderr)
        return rs
    if args.map:
        fmap = FieldMap.from_dict(json.loads(open(args.map).read()))
    else:
        info = inspect_path(args.path)
        fmap = info["guess"]
        print(f"[guessed mapping] {fmap}\n", file=sys.stderr)
    rs, dropped = load(args.path, fmap)
    if dropped:
        print(f"[dropped records] {dropped}\n", file=sys.stderr)
    return rs


def _print_curve(rows: list[dict]) -> None:
    print(f"\n{'k':>3}{'Jaccard vs full':>18}{'Hausdorff':>12}"
          f"{'acc inflation':>16}{'cost inflation':>16}{'|front|':>9}")
    print("-" * 74)
    for r in rows:
        print(
            f"{r['k']:>3}{r['jaccard_vs_full']:>18.3f}{r['hausdorff_vs_full']:>12.4f}"
            f"{r['acc_inflation']:>+16.4f}{r['cost_inflation']:>+16.4f}"
            f"{r['frontier_size']:>9.1f}"
        )
    print("\nacc inflation = what a k-seed study claims for its own frontier winners,")
    print("minus what the full data says about those same configs.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="claimeval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end run on simulated data")
    d.add_argument("--seeds", type=int, default=20)
    d.add_argument("--tasks", type=int, default=50)
    d.add_argument("--configs", type=int, default=9)
    d.add_argument("--boot", type=int, default=2000)

    i = sub.add_parser("inspect", help="discover the schema of a results dump")
    i.add_argument("path")

    a = sub.add_parser("analyze", help="audit frontier claims in a results dump")
    a.add_argument("path")
    a.add_argument("--map", help="JSON field mapping; omit to auto-guess")
    a.add_argument("--from", dest="source", choices=["swebench", "swebench-cost", "terminal-bench", "tau2"],
                   help="read a public leaderboard release directly (see claimeval.adapters)")

    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--threshold", type=float, default=0.90)

    c = sub.add_parser("claims", help="test published claims (top1, pairwise, margin, "
                                      "cost_dominance, frontier) against a results dump")
    c.add_argument("path")
    c.add_argument("claims", help="JSON list of {kind, subject, comparators, margin, id}")
    c.add_argument("--map")
    c.add_argument("--from", dest="source", choices=["swebench", "swebench-cost", "terminal-bench", "tau2"],
                   help="read a public leaderboard release directly (see claimeval.adapters)")

    c.add_argument("--boot", type=int, default=4000)
    c.add_argument("--threshold", type=float, default=0.90)
    c.add_argument("--mode", default=None,
                   choices=["task", "cluster_t"],
                   help="resampling unit; default: clusters if the data has them, else tasks")

    s = sub.add_parser("seed-curve", help="error of a k-seed frontier vs the full data")
    s.add_argument("path")
    s.add_argument("--map")
    s.add_argument("--ks", default="1,2,3,5,10")

    cv = sub.add_parser("coverage", help="validate the estimator against known truth")
    cv.add_argument("--reps", type=int, default=200)
    cv.add_argument("--configs", type=int, default=9)
    cv.add_argument("--tasks", type=int, default=50)
    cv.add_argument("--seeds", type=int, default=5)
    cv.add_argument("--boot", type=int, default=500)
    cv.add_argument("--level", type=float, default=0.95)
    cv.add_argument("--threshold", type=float, default=0.90)
    cv.add_argument("--mode", default="task", choices=["task", "both", "seed"],
                    help="bootstrap mode; 'both' double-counts within-cell noise")

    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    if args.cmd == "demo":
        rolls, truth = simulate(args.configs, args.tasks, args.seeds, rng)
        rs = RolloutSet(rolls)
        report, head = analyze(rs, n_boot=args.boot, rng=rng)
        print(report)
        print("\nGround truth accuracy (simulator):")
        for c, t in zip(truth["configs"], truth["accuracy"]):
            print(f"  {c:<26}{t:.3f}")
        _print_curve(seed_count_curve(rs, (1, 2, 3, 5, 10), n_repeats=30, rng=rng))
        return 0

    if args.cmd == "coverage":
        from claimeval.sim.coverage import coverage_study, format_coverage

        res = coverage_study(
            n_reps=args.reps, n_configs=args.configs, n_tasks=args.tasks,
            n_seeds=args.seeds, n_boot=args.boot, level=args.level,
            threshold=args.threshold, boot_mode=args.mode,
        )
        print(format_coverage(res))
        return 0

    if args.cmd == "inspect":
        info = inspect_path(args.path)
        print(f"{info['n_records']} records scanned\n")
        print(f"{'field':<46}{'count':>8}  samples")
        print("-" * 78)
        for k, n in info["keys"].most_common(60):
            ex = ", ".join(str(v)[:18] for v in info["samples"].get(k, []))
            print(f"{k[:46]:<46}{n:>8}  {ex[:24]}")
        print(f"\nguessed mapping: {info['guess']}")
        print("Verify this, then pass it with --map. Wrong success/cost fields")
        print("produce a confident, meaningless frontier.")
        return 0

    rs = _load(args)
    if args.cmd == "claims":
        from claimeval.analysis.claims import Claim, evaluate_claims

        spec = json.loads(open(args.claims).read())
        cl = [Claim(x["kind"], x["subject"], x.get("comparators", []), x.get("margin", 0.0),
                    x.get("id", ""), x.get("k", 1)) for x in spec]
        res = evaluate_claims(rs, cl, n_boot=args.boot, mode=args.mode, rng=rng, thr=args.threshold)
        print(f"{'id':<10}{'kind':<16}{'subject':<28}{'P(claim)':>9}  verdict")
        for r in res:
            v = "supported" if r["supported"] else (
                "NOT supported" if r["holds_at_point_estimate"] else "contradicted by point estimate")
            print(f"{r['id'][:9]:<10}{r['kind']:<16}{r['subject'][:27]:<28}{r['probability']:>9.3f}  {v}")
        print(f"\nsupported = P >= {args.threshold}; resampling: {res[0]['mode'] if res else '-'}")
        return 0
    if args.cmd == "analyze":
        report, _ = analyze(rs, n_boot=args.boot, threshold=args.threshold, rng=rng)
        print(report)
    else:
        ks = tuple(int(x) for x in args.ks.split(","))
        print(rs.describe())
        _print_curve(seed_count_curve(rs, ks, n_repeats=30, rng=rng))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
