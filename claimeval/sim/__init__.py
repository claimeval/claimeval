"""Synthetic benchmarks, population simulations, and coverage validation."""

from .simulate import Population, make_population, population_truth, sample_dataset, simulate
from .coverage import FAMILIES, Calibration, coverage_study, format_coverage, robustness_sweep

__all__ = [
    "Population",
    "make_population",
    "population_truth",
    "sample_dataset",
    "simulate",
    "FAMILIES",
    "Calibration",
    "coverage_study",
    "format_coverage",
    "robustness_sweep",
]
