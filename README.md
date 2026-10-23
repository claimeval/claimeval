# claimeval

[![PyPI version](https://img.shields.io/pypi/v/claimeval)](https://pypi.org/project/claimeval/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23669069.svg)](https://doi.org/10.5281/zenodo.23669069)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Uncertainty quantification for cost–accuracy claims about AI agents.**

Leaderboards, press releases, and papers declare:
* *"Agent A is #1 on SWE-bench Verified"*
* *"Model X outperforms Model Y"*
* *"System A matches System B at a fifth of the cost"*
* *"Agent A is Pareto-optimal on the cost–accuracy frontier"*

Each of these is a statement about noisy sample estimates from a finite benchmark of tasks and runs. `claimeval` attaches a probability to each such claim, using a bootstrap over the correct unit of generalization, and reports whether it is statistically supported.

---

## 🚀 Quick Start

### Installation

`claimeval` requires only **NumPy**:

```bash
pip install claimeval
```

Or install from source in editable mode:

```bash
git clone https://github.com/claimeval/claimeval.git
cd claimeval
pip install -e .
```

---

## 🔍 Test a Claim in One Command

Evaluate claims against a raw rollout log:

```bash
claimeval claims examples/runs.jsonl examples/claims.json --map examples/map.json
```

**Output:**
```
id        kind            subject                      P(claim)  verdict
c1        top1            agent-E                         0.960  supported
c2        pairwise        agent-D                         0.996  supported
c3        cost_dominance  agent-D                         0.717  NOT supported
c4        frontier        agent-A                         0.993  supported
supported = P >= 0.9; resampling: task
```

### Claim Definitions (`claims.json`)

Define claims in testable JSON schemas:

```json
[
  {
    "id": "c1",
    "kind": "top1",
    "subject": "agent-A",
    "comparators": ["agent-B", "agent-C"]
  },
  {
    "id": "c2",
    "kind": "pairwise",
    "subject": "agent-A",
    "comparators": ["agent-B"]
  },
  {
    "id": "c3",
    "kind": "cost_dominance",
    "subject": "agent-B",
    "comparators": ["agent-D"]
  },
  {
    "id": "c4",
    "kind": "frontier",
    "subject": "agent-A",
    "comparators": ["agent-B", "agent-C", "agent-D"]
  }
]
```

Supported claim types:
* `top1`: The subject strictly leads all comparators.
* `topk`: The subject ranks within the top $k$ systems.
* `pairwise`: The subject strictly outperforms named rivals.
* `parity`: Performance difference is within a threshold ($|a_c - a_f| \le \delta$).
* `cheaper_noninferior`: Subject is cheaper than rival and accuracy is within parity margin.
* `frontier`: Subject belongs to the non-dominated Pareto frontier.
* `cost_dominance`: Subject is both strictly cheaper and more accurate than rival.

---

## 🐍 Python API

```python
from claimeval import Rollout, RolloutSet, Claim, evaluate_claims

# Ingest rollout records: (config, task_id, seed, success_bool, cost_dollars, group)
rollouts = [
    Rollout("agent-A", "task-1", seed=0, success=1.0, cost=0.15, group="repo-1"),
    Rollout("agent-B", "task-1", seed=0, success=0.0, cost=0.08, group="repo-1"),
    # ...
]
dataset = RolloutSet(rollouts)

# Test a pairwise claim
claims = [Claim("pairwise", subject="agent-A", comparators=["agent-B"])]
results = evaluate_claims(dataset, claims, threshold=0.9, n_boot=2000)

for r in results:
    print(f"Claim {r['id']}: P = {r['probability']:.3f} -> {r['verdict']}")
```

---

## 📊 Auditing Public Benchmarks Directly

`claimeval` provides built-in loaders for public agent benchmarks:

```bash
# SWE-bench Verified (from github.com/SWE-bench/experiments)
claimeval claims evaluation/verified claims.json --from swebench

# SWE-bench Verified cost leaderboard (mini-SWE-agent runs)
claimeval claims evaluation/verified claims.json --from swebench-cost

# Terminal-Bench 2.0 (Harbor evaluation trees)
claimeval claims submissions/terminal-bench/2.0 claims.json --from terminal-bench

# tau2-bench leaderboard trajectories
claimeval claims tau2_results/ claims.json --from tau2
```

---

## 🛠️ CLI Toolkit

| Command | Purpose |
|---|---|
| `claimeval claims` | Evaluate a set of typed claims against a rollout dataset |
| `claimeval analyze` | Compute empirical Pareto frontier with membership & dominance probabilities |
| `claimeval inspect` | Automatically inspect schema and statistics of an unknown evaluation dump |
| `claimeval seed-curve` | Simulate error of a $k$-run study vs. full data |
| `claimeval coverage` | Validate estimator calibration against known ground truth |
| `claimeval demo` | Run an end-to-end demonstration on synthetic benchmark data |

---

## 📖 Citation & Research Paper

If you use `claimeval` in your research, please cite our paper:

> **Cost-Accuracy Frontiers for LLM Agents Are Estimated, Not Observed**  
> Harish Mashetty (2026)  
> DOI: [10.5281/zenodo.23669069](https://doi.org/10.5281/zenodo.23669069)

```bibtex
@article{mashetty2026costaccuracy,
  title   = {Cost-Accuracy Frontiers for LLM Agents Are Estimated, Not Observed},
  author  = {Mashetty, Harish},
  year    = {2026},
  doi     = {10.5281/zenodo.23669069},
  url     = {https://doi.org/10.5281/zenodo.23669069}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
