"""Backtest logic checks:
fold order, no look-ahead (the poison test, with a run-once control that
must fail it), the numbering reset at every origin, matched training days,
and the full-view identity (exactly 0 at share 1.0 with no voids).

Trial seeds only; the frozen forecasting seeds are never used here.
"""

import math
from dataclasses import replace

import pandas as pd
import pytest

from ordergaps.backtest import (
    MODELS,
    SERIES,
    build_fold,
    estimated_history,
    forecast_days,
    reset_span_day_indexes,
    rolling_origins,
    score_fold,
)
from ordergaps.config import EstimatorParams
from ordergaps.estimator import estimate
from ordergaps.evaluation.run_all import frozen_base_config
from ordergaps.simulator import SimConfig, simulate

TRIAL_SEED = 404


@pytest.fixture(scope="module")
def frozen_trial_sim():
    return simulate(frozen_base_config(), TRIAL_SEED)


def test_frozen_config_gives_six_weekly_origins():
    origins = rolling_origins(frozen_base_config().n_weeks * 7)
    assert origins == [28, 35, 42, 49, 56, 63]


def test_folds_are_ordered_and_test_weeks_do_not_overlap(frozen_trial_sim):
    previous_last_test_day = -1
    origins = rolling_origins(70)
    for origin in origins:
        fold = build_fold(frozen_trial_sim, origin)
        first_test_day = min(fold.test_days)
        for series in SERIES:
            history = fold.histories[series]
            assert max(history) < first_test_day
        assert fold.test_days == list(range(origin, origin + 7))
        assert first_test_day > previous_last_test_day
        previous_last_test_day = max(fold.test_days)


def test_store_histories_use_matched_training_days(frozen_trial_sim):
    fold = build_fold(frozen_trial_sim, 42)
    est_days = sorted(fold.histories["store_est"])
    true_days = sorted(fold.histories["store_true"])
    assert est_days == true_days
    # The incomplete last day before the origin is dropped from both.
    assert fold.dropped_day == 41
    assert 41 not in fold.histories["store_est"]
    # Your own orders are fully known: every day before the origin.
    assert sorted(fold.histories["vendor"]) == list(range(42))


def fold_fingerprint(fold):
    """Training histories plus every model's forecasts for the fold."""
    forecasts = {}
    for series in SERIES:
        history = fold.histories[series]
        for model in MODELS:
            if model == "oracle":
                continue
            predictions = forecast_days(history, fold.test_days, model, {})
            forecasts[(series, model)] = predictions
    return fold.histories, forecasts


def poison_future(sim, origin):
    """The same simulation with every observed order after the origin changed:
    half of them dropped, the rest renumbered and moved later in the day."""
    start = pd.Timestamp(sim.config.start_day)
    origin_ts = start + pd.Timedelta(days=origin)
    observed = sim.observed.copy()
    future_mask = observed["created_at"] >= origin_ts
    future_index = list(observed.index[future_mask])
    drop_index = future_index[::2]
    poisoned = observed.drop(index=drop_index)
    keep_mask = poisoned["created_at"] >= origin_ts
    poisoned.loc[keep_mask, "order_number"] = poisoned.loc[keep_mask, "order_number"] + 7
    shifted = poisoned.loc[keep_mask, "created_at"] + pd.Timedelta(minutes=30)
    poisoned.loc[keep_mask, "created_at"] = shifted
    poisoned = poisoned.sort_values("created_at", kind="stable").reset_index(drop=True)
    return replace(sim, observed=poisoned)


def run_once_history(sim, origin):
    """A LEAKY history: estimate() once over all the data, then keep the days
    before the origin. Used only as the control the poison test must catch."""
    start = pd.Timestamp(sim.config.start_day)
    start_date = start.date()
    origin_ts = start + pd.Timedelta(days=origin)
    result = estimate(sim.observed, sim.config.store_hours)
    history = {}
    daily_tuples = list(result.daily.itertuples(index=False))
    for row in daily_tuples:
        index = (row.day - start_date).days
        if row.covered and row.day < origin_ts.date():
            history[index] = float(row.est_total)
    return history


def test_no_look_ahead_poisoning_the_future_changes_nothing(frozen_trial_sim):
    origin = 42
    poisoned_sim = poison_future(frozen_trial_sim, origin)
    assert len(poisoned_sim.observed) < len(frozen_trial_sim.observed)
    honest = fold_fingerprint(build_fold(frozen_trial_sim, origin))
    after_poison = fold_fingerprint(build_fold(poisoned_sim, origin))
    assert honest == after_poison


def test_control_run_once_implementation_fails_the_poison_test(frozen_trial_sim):
    origin = 42
    poisoned_sim = poison_future(frozen_trial_sim, origin)
    leaky = run_once_history(frozen_trial_sim, origin)
    leaky_after_poison = run_once_history(poisoned_sim, origin)
    assert leaky != leaky_after_poison


def test_numbering_reset_is_handled_at_every_origin(frozen_trial_sim):
    config = frozen_trial_sim.config
    reset_day = 7 * config.reset_week
    start = pd.Timestamp(config.start_day)
    start_date = start.date()
    truth = {}
    daily_tuples = list(frozen_trial_sim.truth_daily.itertuples(index=False))
    for row in daily_tuples:
        truth[(row.day - start_date).days] = float(row.total)
    origins = rolling_origins(70)
    for origin in origins:
        origin_ts = start + pd.Timedelta(days=origin)
        seen = frozen_trial_sim.observed[frozen_trial_sim.observed["created_at"] < origin_ts]
        history, result, _info = estimated_history(seen, config.store_hours, start_date, origin)
        resets = 0
        for anomaly in result.anomalies:
            if anomaly["kind"] == "numbering_reset":
                resets += 1
        span_days = reset_span_day_indexes(result, start_date)
        if origin > reset_day:
            assert resets == 1
            assert reset_day in span_days
        else:
            assert resets == 0
            assert span_days == set()
        for day in span_days:
            assert day not in history
        history_items = history.items()
        for day, value in history_items:
            assert abs(value - truth[day]) / truth[day] < 0.10


def test_control_without_the_era_split_the_reset_breaks_the_history(frozen_trial_sim):
    config = frozen_trial_sim.config
    start = pd.Timestamp(config.start_day)
    start_date = start.date()
    origin = 49
    origin_ts = start + pd.Timedelta(days=origin)
    seen = frozen_trial_sim.observed[frozen_trial_sim.observed["created_at"] < origin_ts]
    no_split = EstimatorParams(reset_min_drop=10**9)
    result = estimate(seen, config.store_hours, no_split)
    truth = {}
    daily_tuples = list(frozen_trial_sim.truth_daily.itertuples(index=False))
    for row in daily_tuples:
        truth[(row.day - start_date).days] = float(row.total)
    worst = 0.0
    estimate_tuples = list(result.daily.itertuples(index=False))
    for row in estimate_tuples:
        if not row.covered:
            continue
        index = (row.day - start_date).days
        worst = max(worst, abs(row.est_total - truth[index]) / truth[index])
    assert worst >= 0.10


def full_view_config():
    # 7 weeks with a reset at week 4, so the last origin's history crosses it.
    return SimConfig(n_weeks=7, vendor_share=1.0, void_rate=0.0, reset_week=4,
                     drift_per_week=0.01)


def test_full_view_identity_headline_is_exactly_zero():
    config = full_view_config()
    sim = simulate(config, seed=11)
    origins = rolling_origins(config.n_weeks * 7)
    assert origins == [28, 35, 42]
    est_error = 0.0
    true_error = 0.0
    for origin in origins:
        fold = build_fold(sim, origin)
        assert fold.histories["store_est"] == fold.histories["store_true"]
        rows = score_fold(fold, series_names=("store_est", "store_true"),
                          model_names=("weekday_avg",))
        for row in rows:
            if row["series"] == "store_est":
                est_error += row["sum_abs_err"]
            else:
                true_error += row["sum_abs_err"]
    assert true_error > 0.0
    headline = est_error / true_error - 1.0
    assert headline == 0.0


def test_control_unmatched_training_days_break_the_identity():
    # Keeping the incomplete last day on ONE side ("unequal training
    # windows") makes the identity fail even at the full view.
    config = full_view_config()
    sim = simulate(config, seed=11)
    fold = build_fold(sim, 35)
    unmatched = dict(fold.histories["store_true"])
    unmatched[fold.dropped_day] = fold.dropped_true
    honest = forecast_days(fold.histories["store_est"], fold.test_days, "weekday_avg", {})
    skewed = forecast_days(unmatched, fold.test_days, "weekday_avg", {})
    assert honest != skewed


def test_score_fold_rows_are_complete_and_finite(frozen_trial_sim):
    fold = build_fold(frozen_trial_sim, 28)
    rows = score_fold(fold)
    assert len(rows) == len(SERIES) * len(MODELS)
    for row in rows:
        assert row["n_days"] == 7
        assert 0 <= row["n_in80"] <= row["n_in95"] <= 7
        assert math.isfinite(row["mase"])
