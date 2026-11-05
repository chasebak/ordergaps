"""Leak-free rolling-origin backtest (time-series cross-validation).

Folds: a 28-day warm-up, then an origin every 7 days; each fold trains on
days before its origin and forecasts the 7 days from the origin on. Never a
shuffled k-fold: shuffling puts next week's orders into the training set.

No look-ahead, by construction: at EACH origin the estimator is re-run on
the vendor's orders created before the origin only (a single estimate()
over all the data would rewrite past days with future gaps; see
tests/test_backtest.py for the poison test that such a run fails).

Three series per fold, all scored on what really happened:
- store_est: daily store totals as the vendor would estimate them;
- store_true: the same days' TRUE totals (matched training days, so
  store_est vs store_true isolates history quality, not history length);
- vendor: the vendor's own daily orders (directly observed, every day
  before the origin).
"""

import math
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from ordergaps.classify import FULL
from ordergaps.estimator import estimate
from ordergaps.forecast import (
    POINT_FORECASTERS,
    in_sample_naive_mae,
    poisson_interval,
    poisson_mass_between,
    poisson_median,
)
from ordergaps.simulator import true_daily_mean

WARMUP_DAYS = 28
HORIZON_DAYS = 7
STEP_DAYS = 7
LEVELS = (0.80, 0.95)
SERIES = ("store_est", "store_true", "vendor")
MODELS = ("naive", "avg28", "weekday_avg", "oracle")


def rolling_origins(n_days, warmup_days=WARMUP_DAYS, horizon_days=HORIZON_DAYS,
                    step_days=STEP_DAYS):
    """Origin day indexes: warmup_days, then every step_days while a full
    horizon still fits. On 70 days with 28/7/7 that is 6 origins
    (28, 35, 42, 49, 56, 63)."""
    origins = []
    origin = warmup_days
    while origin + horizon_days <= n_days:
        origins.append(origin)
        origin += step_days
    return origins


@dataclass
class Fold:
    """Everything one fold's forecasts may use, plus what they are scored on."""

    seed: int
    share: float
    origin: int
    histories: dict  # series -> {day_index: training value}
    test_days: list
    actuals: dict  # series -> {day_index: actual value}
    oracle_means: dict  # series -> {day_index: true expected count}
    n_full_train_days: int = 0
    n_reset_span_days: int = 0
    dropped_day: int | None = None
    dropped_est: float = math.nan
    dropped_true: float = math.nan
    learned_rates: dict = field(default_factory=dict)


def day_index_of(day, start_date):
    return (day - start_date).days


def reset_span_day_indexes(result, start_date):
    """Day indexes inside any numbering-reset span (span_start .. at).

    The orders between the last pre-reset sighting and the first post-reset
    one are lost to the ID arithmetic, so those days' estimates are short.
    """
    indexes = set()
    one_day = timedelta(days=1)
    for anomaly in result.anomalies:
        if anomaly.get("kind") != "numbering_reset":
            continue
        span_day = anomaly["span_start"].date()
        last_day = anomaly["at"].date()
        while span_day <= last_day:
            indexes.add(day_index_of(span_day, start_date))
            span_day = span_day + one_day
    return indexes


def estimated_history(observed_before_origin, store_hours, start_date, origin):
    """Re-estimate from orders seen before the origin; keep the training days.

    Training days = the estimator's covered days (strictly between its first
    and last seen day, so the incomplete last open day before the origin is
    dropped) minus reset-span days, all before the origin.
    Returns (history, result, info).
    """
    result = estimate(observed_before_origin, store_hours)
    excluded = reset_span_day_indexes(result, start_date)
    history = {}
    n_full = 0
    dropped_day = None
    dropped_est = math.nan
    last_seen_index = day_index_of(result.last_seen.date(), start_date)
    daily_tuples = list(result.daily.itertuples(index=False))
    for row in daily_tuples:
        index = day_index_of(row.day, start_date)
        if index == last_seen_index:
            dropped_day = index
            dropped_est = float(row.est_total)
        if not row.covered or index >= origin or index in excluded:
            continue
        history[index] = float(row.est_total)
        if row.label == FULL:
            n_full += 1
    info = {
        "n_full_train_days": n_full,
        "n_reset_span_days": len(excluded),
        "dropped_day": dropped_day,
        "dropped_est": dropped_est,
    }
    return history, result, info


def true_daily_totals(sim):
    """{day_index: true total orders created that day} from the simulator."""
    start_date = pd.Timestamp(sim.config.start_day).date()
    totals = {}
    daily_tuples = list(sim.truth_daily.itertuples(index=False))
    for row in daily_tuples:
        totals[day_index_of(row.day, start_date)] = float(row.total)
    return totals


def vendor_daily_counts(observed, start_date, n_days):
    """{day_index: the vendor's own orders that day} for days 0..n_days-1."""
    counts = {}
    day_values = range(n_days)
    for index in day_values:
        counts[index] = 0.0
    created_values = list(observed["created_at"])
    for created_at in created_values:
        index = day_index_of(created_at.date(), start_date)
        if 0 <= index < n_days:
            counts[index] += 1.0
    return counts


def build_fold(sim, origin, horizon_days=HORIZON_DAYS):
    """One fold: histories from data before the origin, actuals after it."""
    config = sim.config
    start = pd.Timestamp(config.start_day)
    start_date = start.date()
    origin_ts = start + pd.Timedelta(days=origin)
    seen_before = sim.observed[sim.observed["created_at"] < origin_ts]
    est_history, result, info = estimated_history(
        seen_before, config.store_hours, start_date, origin
    )

    truth = true_daily_totals(sim)
    true_history = {}
    est_days = sorted(est_history)
    for index in est_days:
        true_history[index] = truth[index]

    vendor_history = vendor_daily_counts(seen_before, start_date, origin)

    test_days = []
    day_offsets = range(horizon_days)
    for offset in day_offsets:
        test_days.append(origin + offset)
    vendor_all = vendor_daily_counts(sim.observed, start_date, origin + horizon_days)
    vendor_factor = config.vendor_share * (1.0 - config.void_rate)
    store_actuals = {}
    vendor_actuals = {}
    store_means = {}
    vendor_means = {}
    for index in test_days:
        store_actuals[index] = truth[index]
        vendor_actuals[index] = vendor_all[index]
        mean_total = true_daily_mean(config, index)
        store_means[index] = mean_total
        vendor_means[index] = mean_total * vendor_factor

    dropped_day = info["dropped_day"]
    dropped_true = math.nan
    if dropped_day is not None:
        dropped_true = truth[dropped_day]

    return Fold(
        seed=sim.seed,
        share=config.vendor_share,
        origin=origin,
        histories={
            "store_est": est_history,
            "store_true": true_history,
            "vendor": vendor_history,
        },
        test_days=test_days,
        actuals={
            "store_est": store_actuals,
            "store_true": store_actuals,
            "vendor": vendor_actuals,
        },
        oracle_means={
            "store_est": store_means,
            "store_true": store_means,
            "vendor": vendor_means,
        },
        n_full_train_days=info["n_full_train_days"],
        n_reset_span_days=info["n_reset_span_days"],
        dropped_day=dropped_day,
        dropped_est=info["dropped_est"],
        dropped_true=dropped_true,
        learned_rates=dict(result.rates.per_weekday),
    )


def forecast_days(history, test_days, model, oracle_means):
    """[(point, interval_mean)] per test day for one model.

    Point = the forecast (MAE is scored on it); the oracle's point is its
    Poisson median (MAE-optimal). Intervals are Poisson around interval_mean.
    """
    predictions = []
    for day in test_days:
        if model == "oracle":
            mean = oracle_means[day]
            predictions.append((float(poisson_median(mean)), mean))
            continue
        forecaster = POINT_FORECASTERS[model]
        mean = forecaster(history, day)
        if math.isnan(mean):
            raise ValueError(f"{model} has no history for day {day}")
        predictions.append((mean, mean))
    return predictions


def score_predictions(predictions, test_days, actuals, oracle_means, scale):
    """Sums for one (fold, series, model); pooled metrics come from these.

    - abs error and actual sums (MAE, WAPE); MASE = fold MAE / scale, where
      scale = in-sample seasonal-naive MAE on the training days;
    - interval hits at 80% and 95%, and the 80% width;
    - achievable coverage: each day's exact Poisson chance of landing in
      the ORACLE's interval, and its p(1 - p) for the Poisson-binomial SE.
    """
    n_days = len(test_days)
    sum_abs_err = 0.0
    sum_actual = 0.0
    hits = {}
    achievable = {}
    achievable_var = {}
    for level in LEVELS:
        hits[level] = 0
        achievable[level] = 0.0
        achievable_var[level] = 0.0
    sum_width80 = 0.0
    n_zero_actual = 0
    day_positions = range(n_days)
    for position in day_positions:
        day = test_days[position]
        point, interval_mean = predictions[position]
        actual = actuals[day]
        true_mean = oracle_means[day]
        sum_abs_err += abs(point - actual)
        sum_actual += actual
        if actual == 0:
            n_zero_actual += 1
        for level in LEVELS:
            low, high = poisson_interval(interval_mean, level)
            if low <= actual <= high:
                hits[level] += 1
            if level == 0.80:
                sum_width80 += high - low
            oracle_low, oracle_high = poisson_interval(true_mean, level)
            probability = poisson_mass_between(true_mean, oracle_low, oracle_high)
            achievable[level] += probability
            achievable_var[level] += probability * (1.0 - probability)
    mae = sum_abs_err / n_days
    if scale > 0.0:
        mase = mae / scale
    else:
        mase = math.nan
    return {
        "n_days": n_days,
        "sum_abs_err": sum_abs_err,
        "sum_actual": sum_actual,
        "mae": mae,
        "scale": scale,
        "mase": mase,
        "n_in80": hits[0.80],
        "n_in95": hits[0.95],
        "sum_width80": sum_width80,
        "n_zero_actual": n_zero_actual,
        "achievable80_sum": achievable[0.80],
        "achievable95_sum": achievable[0.95],
        "achievable80_var": achievable_var[0.80],
        "achievable95_var": achievable_var[0.95],
    }


def score_fold(fold, series_names=SERIES, model_names=MODELS):
    """One score row per (series, model) of a fold."""
    rows = []
    for series in series_names:
        history = fold.histories[series]
        scale = in_sample_naive_mae(history)
        actuals = fold.actuals[series]
        means = fold.oracle_means[series]
        for model in model_names:
            predictions = forecast_days(history, fold.test_days, model, means)
            scores = score_predictions(predictions, fold.test_days, actuals, means, scale)
            row = {
                "share": fold.share,
                "seed": fold.seed,
                "origin": fold.origin,
                "series": series,
                "model": model,
            }
            row.update(scores)
            rows.append(row)
    return rows
