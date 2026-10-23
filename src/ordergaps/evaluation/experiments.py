"""The three evaluation experiments against simulator ground truth.

1. Error vs fraction of orders observed (hero): across seeds and vendor
   shares, median |relative error| of weekly store volume vs truth.
2. Length bias: the weekday-rate estimator learns only from gaps fully
   inside sessions, and observation times are the vendor's own sales, so
   busy periods are oversampled; bias of learned rates vs true rates.
3. Counted vs allocated: near-exact quantities vs model-based ones.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ordergaps.classify import FULL, PARTIAL, SPREAD
from ordergaps.estimator import estimate
from ordergaps.simulator import simulate, with_share


@dataclass
class RunMetrics:
    share: float
    seed: int
    fraction_observed: float
    n_weeks_used: int
    weekly_err_median: float  # median |rel err| of weekly store totals
    channel_err_median: float  # median |rel err| of weekly in-store totals
    rate_bias: float  # minutes-weighted mean relative bias of learned rates
    n_windows: int
    n_windows_exact: int
    full_day_share_abs_err: float  # |vendor share of all orders on full days - true share|
    full_day_err_median: float  # per-day TOTAL volume |rel err|, full days
    nonfull_day_err_median: float  # per-day TOTAL volume |rel err|, partial+spread days


def _median(values):
    if not values:
        return float("nan")
    return float(np.median(values))


def run_one(config, seed):
    """Simulate one regime, estimate, and score against ground truth."""
    sim = simulate(config, seed)
    result = estimate(sim.observed, config.store_hours)

    weekly = result.weekly
    clean = weekly[weekly["fully_covered"] & ~weekly["has_anomaly"]]
    merged = clean.merge(sim.truth_weekly, on=["iso_year", "iso_week"], suffixes=("", "_true"))
    weekly_errors = []
    channel_errors = []
    merged_tuples = list(merged.itertuples(index=False))
    for row in merged_tuples:
        if row.total > 0:
            weekly_errors.append(abs(row.est_total - row.total) / row.total)
        if row.in_store > 0:
            channel_errors.append(abs(row.est_in_store - row.in_store) / row.in_store)

    bias_weighted_sum = 0.0
    bias_weight_total = 0.0
    weekday_values = range(7)
    for weekday in weekday_values:
        true_rate = sim.true_weekday_rates.get(weekday)
        if true_rate is None or true_rate <= 0.0:
            continue
        scheduled = config.store_hours.for_weekday(weekday)
        if scheduled is None:
            continue
        open_minute, close_minute = scheduled
        weight = float(close_minute - open_minute)
        learned_rate = result.rates.rate_for(weekday)
        bias = (learned_rate - true_rate) / true_rate
        bias_weighted_sum += bias * weight
        bias_weight_total += weight
    if bias_weight_total > 0.0:
        rate_bias = bias_weighted_sum / bias_weight_total
    else:
        rate_bias = float("nan")

    truth_times = np.sort(sim.orders["created_at"].to_numpy())
    n_windows = 0
    n_windows_exact = 0
    for gap in result.gaps:
        t1 = gap.t1.to_datetime64()
        t2 = gap.t2.to_datetime64()
        count_before_t2 = int(np.searchsorted(truth_times, t2, side="left"))
        count_through_t1 = int(np.searchsorted(truth_times, t1, side="right"))
        between = count_before_t2 - count_through_t1
        n_windows += 1
        if between == gap.unseen:
            n_windows_exact += 1

    # Full days: the day's TOTAL volume is a counted quantity (boundary spill
    # only); vendor share over those pinned windows is sampling noise only.
    # Partial/spread days get their totals from the timing model.
    daily = result.daily.merge(sim.truth_daily, on="day", suffixes=("", "_true"))
    full_vendor_orders = 0
    full_est_total = 0.0
    full_errors = []
    nonfull_errors = []
    daily_tuples = list(daily.itertuples(index=False))
    for row in daily_tuples:
        if row.label == FULL:
            full_vendor_orders += row.vendor_orders
            full_est_total += row.est_total
            if row.total > 0:
                full_errors.append(abs(row.est_total - row.total) / row.total)
        elif row.label in (PARTIAL, SPREAD):
            if row.total > 0:
                nonfull_errors.append(abs(row.est_total - row.total) / row.total)
    if full_est_total > 0.0:
        full_day_share = full_vendor_orders / full_est_total
        full_day_share_abs_err = abs(full_day_share - config.vendor_share)
    else:
        full_day_share_abs_err = float("nan")

    return RunMetrics(
        share=config.vendor_share,
        seed=seed,
        fraction_observed=sim.fraction_observed,
        n_weeks_used=len(merged),
        weekly_err_median=_median(weekly_errors),
        channel_err_median=_median(channel_errors),
        rate_bias=rate_bias,
        n_windows=n_windows,
        n_windows_exact=n_windows_exact,
        full_day_share_abs_err=full_day_share_abs_err,
        full_day_err_median=_median(full_errors),
        nonfull_day_err_median=_median(nonfull_errors),
    )


def run_matrix(base_config, shares, seeds):
    """All (share, seed) runs as a DataFrame of RunMetrics rows."""
    records = []
    for share in shares:
        config = with_share(base_config, share)
        for seed in seeds:
            metrics = run_one(config, seed)
            records.append(metrics.__dict__)
    return pd.DataFrame(records)


def rank_with_ties(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        tied_positions = range(i, j + 1)
        for k in tied_positions:
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def spearman(x, y):
    x_ranks = rank_with_ties(x)
    y_ranks = rank_with_ties(y)
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = math.sqrt(float(np.sum(x_centered**2)) * float(np.sum(y_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered)) / denominator


def count_inversions(errors_by_increasing_fraction):
    """Consecutive places where error INCREASES as the observed fraction grows."""
    inversions = 0
    pair_indexes = range(1, len(errors_by_increasing_fraction))
    for index in pair_indexes:
        previous_error = errors_by_increasing_fraction[index - 1]
        current_error = errors_by_increasing_fraction[index]
        if current_error > previous_error:
            inversions += 1
    return inversions


def error_curve(records):
    """Per share: mean observed fraction, median across seeds of the per-run
    median weekly error."""
    rows = []
    share_values = sorted(records["share"].unique())
    for share in share_values:
        share_rows = records[records["share"] == share]
        rows.append(
            {
                "share": share,
                "mean_fraction_observed": float(share_rows["fraction_observed"].mean()),
                "median_weekly_err": float(share_rows["weekly_err_median"].median()),
            }
        )
    curve = pd.DataFrame(rows)
    return curve.sort_values("mean_fraction_observed").reset_index(drop=True)


def acceptance_check(curve):
    """The frozen acceptance band for experiment 1.

    Spearman rho(fraction observed, median |error|) <= -0.8 with at most
    one inversion along the share-ordered curve.
    """
    rho = spearman(
        curve["mean_fraction_observed"].to_numpy(), curve["median_weekly_err"].to_numpy()
    )
    inversions = count_inversions(list(curve["median_weekly_err"]))
    passed = rho <= -0.8 and inversions <= 1
    return {"spearman": rho, "inversions": inversions, "passed": passed}


def length_bias_summary(records):
    """Per share: mean and spread of the per-run learned-rate bias."""
    rows = []
    share_values = sorted(records["share"].unique())
    for share in share_values:
        share_rows = records[records["share"] == share]
        biases = share_rows["rate_bias"].to_numpy(dtype=float)
        rows.append(
            {
                "share": share,
                "mean_fraction_observed": float(share_rows["fraction_observed"].mean()),
                "mean_bias": float(np.nanmean(biases)),
                "p25_bias": float(np.nanpercentile(biases, 25)),
                "p75_bias": float(np.nanpercentile(biases, 75)),
            }
        )
    return pd.DataFrame(rows)


def counted_vs_allocated(records, table_share):
    """Counted (near-exact) vs allocated (model-based) quantities at one share.

    Error bars (IQR across seeds) appear ONLY on the model-based rows;
    counted rows carry exact or sampling-only numbers.
    """
    at_share = records[records["share"] == table_share]
    n_windows = int(at_share["n_windows"].sum())
    n_exact = int(at_share["n_windows_exact"].sum())
    share_errors = at_share["full_day_share_abs_err"].to_numpy(dtype=float)
    full_day = at_share["full_day_err_median"].to_numpy(dtype=float)
    nonfull_day = at_share["nonfull_day_err_median"].to_numpy(dtype=float)
    channel = at_share["channel_err_median"].to_numpy(dtype=float)
    summary = {
        "table_share": table_share,
        "n_windows": n_windows,
        "n_windows_exact": n_exact,
        "full_day_share_mean_abs_err": float(np.nanmean(share_errors)),
        "full_day_err_median": float(np.nanmedian(full_day)),
        "channel_err_median": float(np.nanmedian(channel)),
        "channel_err_p25": float(np.nanpercentile(channel, 25)),
        "channel_err_p75": float(np.nanpercentile(channel, 75)),
        "nonfull_day_err_median": float(np.nanmedian(nonfull_day)),
        "nonfull_day_err_p25": float(np.nanpercentile(nonfull_day, 25)),
        "nonfull_day_err_p75": float(np.nanpercentile(nonfull_day, 75)),
    }
    return summary
