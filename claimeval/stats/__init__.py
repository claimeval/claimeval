"""Statistical estimation, resampling, bootstrap methods, Pareto geometry, and diagnostics."""

from .frontier import dominance, frontier_band, frontier_points, hausdorff, log_cost_range, pareto_mask
from .bootstrap import (
    BootstrapResult,
    Mode,
    auto_mode,
    hierarchical_bootstrap,
    variance_decomposition,
)
from .wild import wild_intervals
from .bca import bca_intervals
from .calibrate import calibrated_frontier_prob, calibration_map
from .bias import seed_count_curve, split_sample_debias
from .cost_axis import diagnose_cost_axis, format_diagnosis

__all__ = [
    "dominance",
    "frontier_band",
    "frontier_points",
    "hausdorff",
    "log_cost_range",
    "pareto_mask",
    "BootstrapResult",
    "Mode",
    "auto_mode",
    "hierarchical_bootstrap",
    "variance_decomposition",
    "wild_intervals",
    "bca_intervals",
    "calibrated_frontier_prob",
    "calibration_map",
    "seed_count_curve",
    "split_sample_debias",
    "diagnose_cost_axis",
    "format_diagnosis",
]
