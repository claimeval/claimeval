"""Which cost axis is safe for *this* dataset?

The coverage study found that neither choice is universally safe, in opposite
regimes:

  - **mean cost** can fail to have a population value at all. Cost scales
    roughly exponentially with task difficulty, so polynomial difficulty tails
    make E[cost] infinite (no MGF). A finite sample still reports a
    stable-looking number. Measured: MC estimates swinging 27 orders of
    magnitude between batches under t(3) difficulty.
  - **median cost** always exists, but the median of *noisy* per-task means is
    a biased estimate of the median of true per-task means. The bias is driven
    by within-cell cost dispersion over seeds, and shrinks with seeds, not with
    tasks — so more tasks narrows the interval around a still-wrong centre.
    Measured coverage under heavy dispersion: 0.478 at 5 seeds, 0.644 at 20,
    0.835 at 60, against a nominal 0.95.

So: report both, and check which regime you are in.

**Asymmetry worth knowing.** The two pathologies are not equally detectable.
Median bias is diagnosable from the data — within-cell dispersion over seeds
separates the regimes cleanly (risk 0.30 in the bad case versus ~0.10 in the
good one). Mean divergence is *not*: at 80 tasks no statistic tried
(mean/median ratio, top-5% share, Hill tail index, max/median) distinguished
t(3)-driven difficulty from lognormal, and max/median only began to separate at
400 tasks — larger than most agent suites. A non-firing tail warning therefore
does not certify mean cost. That asymmetry is why median is the default: its
failure mode announces itself and is fixable with seeds, while the mean's does
not and is not.
"""

from __future__ import annotations

import warnings

import numpy as np

from claimeval.core.schema import RolloutSet

# Thresholds are calibrated against the coverage sweep, not derived. They are
# tripwires for "look closer", not decision procedures.
MEDIAN_BIAS_WARN = 0.15  # within-cell log10 dispersion / sqrt(seeds)
MEAN_TAIL_WARN = 2.0  # across-task mean/median ratio of per-task cost


def diagnose_cost_axis(rs: RolloutSet) -> dict:
    """Flag whether mean cost, median cost, or both are unreliable here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logc = np.log10(np.where(rs.cost > 0, rs.cost, np.nan))
        # within-cell dispersion over seeds, pooled across cells with >= 2 seeds
        disp_cells = np.nanstd(logc, axis=2, ddof=1)
        disp = float(np.nanmedian(disp_cells[rs.counts >= 2])) if (rs.counts >= 2).any() else 0.0

        cost_t = np.nanmean(rs.cost, axis=2)  # (C, T) per-task mean cost
        means = np.nanmean(cost_t, axis=1)
        medians = np.nanmedian(cost_t, axis=1)
        ratios = np.divide(means, medians, out=np.full_like(means, np.nan), where=medians > 0)
        tail_ratio = float(np.nanmax(ratios)) if np.isfinite(ratios).any() else np.nan

    seeds = float(np.median(rs.counts[rs.counts > 0])) if (rs.counts > 0).any() else 1.0
    bias_risk = disp / np.sqrt(max(seeds, 1.0))

    notes: list[str] = []
    if bias_risk > MEDIAN_BIAS_WARN:
        notes.append(
            f"median cost is likely biased: within-cell log10 cost dispersion "
            f"{disp:.2f} over ~{seeds:.0f} seeds gives risk {bias_risk:.2f} "
            f"(> {MEDIAN_BIAS_WARN}). More seeds is the only fix; more tasks makes it worse "
            f"by narrowing the interval around a biased centre."
        )
    if tail_ratio > MEAN_TAIL_WARN:
        notes.append(
            f"mean cost is tail-dominated: across-task mean/median ratio {tail_ratio:.1f} "
            f"(> {MEAN_TAIL_WARN}). If task difficulty has polynomial tails the population "
            f"mean may not exist at all; treat any mean-cost frontier as unsupported."
        )
    if not notes:
        notes.append(
            "no cost-axis pathology detected in-sample; mean and median should broadly "
            "agree. Note this does not certify the mean — difficulty-tail divergence is "
            "not detectable at typical suite sizes, so median remains the safer default."
        )

    if bias_risk > MEDIAN_BIAS_WARN and tail_ratio > MEAN_TAIL_WARN:
        recommend = "neither"
    elif tail_ratio > MEAN_TAIL_WARN:
        recommend = "median"
    elif bias_risk > MEDIAN_BIAS_WARN:
        recommend = "mean"
    else:
        recommend = "either"

    return {
        "within_cell_log_dispersion": disp,
        "median_seeds": seeds,
        "median_bias_risk": bias_risk,
        "mean_tail_ratio": tail_ratio,
        "recommend": recommend,
        "notes": notes,
    }


def format_diagnosis(d: dict) -> str:
    L = [
        f"within-cell log10 cost dispersion : {d['within_cell_log_dispersion']:.3f} "
        f"over ~{d['median_seeds']:.0f} seeds",
        f"median-bias risk                  : {d['median_bias_risk']:.3f} "
        f"(warn > {MEDIAN_BIAS_WARN})",
        f"mean tail ratio (mean/median)     : {d['mean_tail_ratio']:.2f} "
        f"(warn > {MEAN_TAIL_WARN})",
        f"safe cost axis                    : {d['recommend']}",
    ]
    L += [f"  - {n}" for n in d["notes"]]
    if d["recommend"] == "neither":
        L.append(
            "  - both axes are compromised here: add seeds before reporting any "
            "cost-accuracy frontier from this data."
        )
    return "\n".join(L)
