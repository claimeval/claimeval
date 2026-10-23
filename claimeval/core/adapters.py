"""Readers for public leaderboard releases, so claims can be checked against them directly.

    swebench(path)       a split directory of github.com/SWE-bench/experiments,
                         e.g. evaluation/verified (one run per task; repository = cluster)
    terminal_bench(path) a Harbor results tree, e.g. the Terminal-Bench 2.0 leaderboard
                         dataset's submissions/terminal-bench/2.0 (several runs per task)
    tau2(files)          tau2-bench results JSON files (several trials per task)

Each returns a RolloutSet and a dict of submissions it skipped, with the reason.
Only per-task outcomes the source itself records are used; nothing is imputed.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from .schema import Rollout, RolloutSet


def _repo(instance_id: str) -> str:
    """'django__django-11099' -> 'django__django' (SWE-bench instance ids encode the repo)."""
    return instance_id.rsplit("-", 1)[0]


def swebench(path: str | Path, with_cost: bool = False) -> tuple[RolloutSet, dict]:
    """Per-task resolved/unresolved for every submission under a SWE-bench experiments split.

    The task universe is the union of instance ids any submission reports, so a
    task a submission did not list counts as unresolved, the leaderboard rule.
    With `with_cost`, only submissions shipping per_instance_details.json with a
    numeric cost for every task are kept (the bash-only / mini-SWE-agent board).
    """
    path = Path(path)
    subs, skipped, universe = {}, {}, set()
    for d in sorted(p for p in path.iterdir() if p.is_dir()):
        if with_cost:
            f = d / "per_instance_details.json"
            if not f.exists():
                continue
            det = json.loads(f.read_text())
            costs = {k: v.get("cost") for k, v in det.items()}
            if not all(isinstance(c, (int, float)) for c in costs.values()):
                skipped[d.name] = "per-task cost missing for some tasks"
                continue
            subs[d.name] = ({k for k, v in det.items() if v.get("resolved")}, costs)
            universe |= set(det)
            continue
        f = d / "results" / "results.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        if not isinstance(res.get("resolved"), list):
            skipped[d.name] = "results.json records only a count, not task ids"
            continue
        for v in res.values():
            if isinstance(v, list):
                universe |= set(v)
        subs[d.name] = (set(res["resolved"]), None)
    rolls = []
    for name, (solved, costs) in subs.items():
        if costs is not None and set(costs) != universe:
            skipped[name] = f"covers {len(costs)} of {len(universe)} tasks"
            continue
        for t in sorted(universe):
            rolls.append(Rollout(name, t, "0", float(t in solved),
                                 float(costs[t]) if costs else 1.0, _repo(t)))
    if not rolls:
        raise ValueError(f"no usable submissions under {path}")
    return RolloutSet(rolls), skipped


def terminal_bench(path: str | Path) -> tuple[RolloutSet, dict]:
    """Harbor result trees: <submission>/<job>/<trial>/result.json.

    A trial with no verifier reward counts as a failure (the harness's own rule).
    Jobs whose trials do not all record the same task-suite commit are skipped,
    since results from different suite versions are not comparable. Cost is the
    trial's recorded `cost_usd`; trials without it get NaN cost, so cost claims
    can only be tested on submissions that record it.
    """
    path = Path(path)
    rolls, skipped, commits = [], {}, {}
    for job in sorted(glob.glob(str(path / "*" / "*" / "result.json"))):
        jdir = os.path.dirname(job)
        sub = os.path.basename(os.path.dirname(jdir))
        trials = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(jdir, "*", "result.json")))]
        if not trials:
            continue
        cs = {(t.get("task_id") or {}).get("git_commit_id") if isinstance(t.get("task_id"), dict)
              else None for t in trials}
        if len(cs) != 1 or None in cs:
            skipped[f"{sub}/{os.path.basename(jdir)}"] = "task-suite commit missing or mixed"
            continue
        commits.setdefault(next(iter(cs)), []).append(sub)
        for i, t in enumerate(trials):
            vr = t.get("verifier_result") or {}
            rew = ((vr.get("rewards") or {}) if isinstance(vr, dict) else {}).get("reward")
            cost = (t.get("agent_result") or {}).get("cost_usd")
            rolls.append(Rollout(sub, (t.get("task_name") or "").split("/")[-1],
                                 f"{os.path.basename(jdir)}/{i}", float(rew or 0.0),
                                 float(cost) if isinstance(cost, (int, float)) else float("nan")))
    if len(commits) > 1:  # keep the suite version most submissions ran
        main = max(commits, key=lambda c: len(commits[c]))
        drop = {s for c, ss in commits.items() if c != main for s in ss} - set(commits[main])
        rolls = [r for r in rolls if r.config not in drop]
        skipped.update({s: "different task-suite version" for s in drop})
    if not rolls:
        raise ValueError(f"no usable trials under {path}")
    return RolloutSet(rolls), skipped


def tau2(files, use_user_cost: bool = False) -> tuple[RolloutSet, dict]:
    """tau2-bench results files (one per submission and domain).

    Configuration name: the file name without extension. Cost is the agent's
    own `agent_cost` (the user simulator's cost is evaluation overhead) unless
    `use_user_cost`. Submissions recording zero cost on every trial are kept for
    accuracy but flagged, since their cost was not logged.
    """
    rolls, skipped = [], {}
    for f in files:
        name = Path(f).stem
        sims = json.loads(Path(f).read_text()).get("simulations") or []
        if not sims:
            skipped[name] = "no simulations in file"
            continue
        costs = []
        for s in sims:
            rew = (s.get("reward_info") or {}).get("reward")
            if rew is None:
                continue
            c = (s.get("agent_cost") or 0.0) + (s.get("user_cost") or 0.0 if use_user_cost else 0.0)
            costs.append(c)
            rolls.append(Rollout(name, str(s.get("task_id")), str(s.get("trial")), float(rew), float(c)))
        if costs and max(costs) == 0:
            skipped[name] = "cost not logged (zero on every trial); accuracy only"
    if not rolls:
        raise ValueError("no scored simulations in the given files")
    return RolloutSet(rolls), skipped
