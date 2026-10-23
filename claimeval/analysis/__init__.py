"""Claim evaluation, comparison, and reporting."""

from claimeval.stats.frontier import dominance, frontier_band, frontier_points, hausdorff, log_cost_range, pareto_mask
from .claims import Claim, evaluate_claims
from .compare import Scenario, compare, format_rows
from .baselines import METHODS, run_method
from .report import analyze, format_report, headline

__all__ = [
    "Claim",
    "evaluate_claims",
    "dominance",
    "frontier_band",
    "frontier_points",
    "hausdorff",
    "log_cost_range",
    "pareto_mask",
    "Scenario",
    "compare",
    "format_rows",
    "METHODS",
    "run_method",
    "analyze",
    "format_report",
    "headline",
]
