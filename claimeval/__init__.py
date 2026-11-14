"""claimeval -- test the claims made about AI agents against the data behind them.

A claim ("#1", "beats X", "as good at lower cost", "on the cost-accuracy
frontier") gets a bootstrap support proportion: the fraction of resamples in
which it holds. This is not a posterior probability that the claim is true.
Input is a
stream of (config, task, seed, success, cost[, group]) runs from any harness,
or a public leaderboard release read by `claimeval.core.adapters`. Also: frontier
membership and dominance probabilities, a Hausdorff frontier band, a
winner's-curse correction, wild cluster bootstrap intervals, and a task-vs-seed
variance decomposition.
"""

import sys
from importlib import import_module

# Subpackages
from . import core
from . import stats
from . import analysis
from . import sim

# Core data structures and loaders
from .core.schema import Rollout, RolloutSet
from .core.ingest import FieldMap, guess_map, inspect_path, load

# Statistical methods, frontier geometry & diagnostics
from .stats.frontier import dominance, frontier_band, frontier_points, hausdorff, log_cost_range, pareto_mask
from .stats.bootstrap import BootstrapResult, auto_mode, hierarchical_bootstrap, variance_decomposition
from .stats.wild import wild_intervals
from .stats.bias import seed_count_curve, split_sample_debias
from .stats.cost_axis import diagnose_cost_axis, format_diagnosis

# Analysis, claims & reporting
from .analysis.claims import Claim, evaluate_claims
from .analysis.report import analyze, format_report, headline

# Simulation & coverage validation
from .sim.simulate import make_population, population_truth, sample_dataset, simulate
from .sim.coverage import coverage_study, format_coverage

# Backward compatibility aliases for module paths
sys.modules["claimeval.schema"] = core.schema
sys.modules["claimeval.ingest"] = core.ingest
sys.modules["claimeval.adapters"] = core.adapters
sys.modules["claimeval.bootstrap"] = stats.bootstrap
sys.modules["claimeval.wild"] = stats.wild
sys.modules["claimeval.bca"] = stats.bca
sys.modules["claimeval.calibrate"] = stats.calibrate
sys.modules["claimeval.bias"] = stats.bias
sys.modules["claimeval.cost_axis"] = stats.cost_axis
sys.modules["claimeval.frontier"] = stats.frontier
sys.modules["claimeval.claims"] = analysis.claims
sys.modules["claimeval.compare"] = import_module(".analysis.compare", __name__)
sys.modules["claimeval.baselines"] = analysis.baselines
sys.modules["claimeval.report"] = analysis.report
sys.modules["claimeval.simulate"] = import_module(".sim.simulate", __name__)
sys.modules["claimeval.coverage"] = import_module(".sim.coverage", __name__)

__version__ = "1.0.2"

__all__ = [
    "core",
    "stats",
    "analysis",
    "sim",
    "Claim",
    "evaluate_claims",
    "auto_mode",
    "wild_intervals",
    "Rollout",
    "RolloutSet",
    "FieldMap",
    "guess_map",
    "inspect_path",
    "load",
    "analyze",
    "headline",
    "format_report",
    "hierarchical_bootstrap",
    "BootstrapResult",
    "variance_decomposition",
    "pareto_mask",
    "dominance",
    "frontier_points",
    "hausdorff",
    "frontier_band",
    "log_cost_range",
    "split_sample_debias",
    "seed_count_curve",
    "simulate",
    "make_population",
    "sample_dataset",
    "population_truth",
    "coverage_study",
    "format_coverage",
]
