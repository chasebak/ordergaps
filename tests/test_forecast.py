"""Forecast functions: hand-computed fixtures, the oracle's parity and
calibration, and controls that must fail.

Seeds here are TRIAL seeds (101..808 and small ones), never the frozen
forecasting seeds 1101..1808, so no test tunes against the frozen run.
"""

import math

from ordergaps.backtest import forecast_days
from ordergaps.forecast import (
    in_sample_naive_mae,
    moving_average,
    poisson_interval,
    poisson_mass_between,
    poisson_median,
    poisson_quantile,
    poisson_quantiles,
    seasonal_naive,
    weekday_average,
)
from ordergaps.simulator import SimConfig, simulate, true_daily_mean

TRIAL_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)


def weekly_series(n_days, profile):
    series = {}
    day_values = range(n_days)
    for day in day_values:
        series[day] = float(profile[day % 7])
    return series


# --- Poisson quantiles and intervals -------------------------------------


def test_poisson_quantile_hand_fixtures():
    # Poisson(10): CDF(3) = 0.0103 < 0.025 <= CDF(4) = 0.0293, and
    # CDF(16) = 0.9730 < 0.975 <= CDF(17) = 0.9857.
    assert poisson_quantile(10, 0.025) == 4
    assert poisson_quantile(10, 0.975) == 17


def test_one_loop_quantiles_equal_single_calls():
    probabilities = [0.975, 0.1, 0.5, 0.025, 0.9]
    together = poisson_quantiles(240.0, probabilities)
    separate = []
    for probability in probabilities:
        separate.append(poisson_quantile(240.0, probability))
    assert together == separate
    assert poisson_median(240.0) == 240


def test_zero_mean_is_a_point_mass():
    assert poisson_interval(0.0, 0.95) == (0, 0)
    assert poisson_mass_between(0.0, 0, 0) == 1.0


def test_exact_80_coverage_reference_values():
    # Whole-number counts over-cover: exact Poisson reference values.
    expected = {1.0: 0.920, 10.0: 0.849, 30.0: 0.830}
    expected_items = expected.items()
    for mean, coverage in expected_items:
        low, high = poisson_interval(mean, 0.80)
        mass = poisson_mass_between(mean, low, high)
        assert round(mass, 3) == coverage


def test_mass_between_sums_to_one_over_the_support():
    total = poisson_mass_between(12.5, 0, 200)
    assert math.isclose(total, 1.0, rel_tol=1e-12)


# --- point forecasters ----------------------------------------------------


def test_seasonal_naive_has_zero_error_on_an_exactly_weekly_series():
    profile = (90, 240, 240, 250, 290, 370, 220)
    history = weekly_series(28, profile)
    test_days = range(28, 35)
    for day in test_days:
        assert seasonal_naive(history, day) == profile[day % 7]
    assert in_sample_naive_mae(history) == 0.0


def test_seasonal_naive_falls_back_to_t_minus_14():
    history = weekly_series(28, (1, 2, 3, 4, 5, 6, 7))
    history[20] = 100.0  # day 20 is the weekday of day 27 two weeks back
    del history[27]  # the dropped incomplete last day
    assert seasonal_naive(history, 34) == 100.0
    assert math.isnan(seasonal_naive({}, 5))


def test_weekday_average_matches_hand_mean():
    history = {0: 10.0, 7: 20.0, 14: 30.0, 21: 40.0, 28: 50.0, 3: 999.0}
    # Last 4 same-weekday values before day 35: 50, 40, 30, 20.
    assert weekday_average(history, 35) == 35.0
    del history[21]
    # A missing week is skipped: 50, 30, 20, 10.
    assert weekday_average(history, 35) == 27.5


def test_moving_average_uses_the_last_28_values_only():
    history = {}
    day_values = range(40)
    for day in day_values:
        history[day] = float(day)
    # Days 12..39 -> mean 25.5, whatever the target weekday.
    assert moving_average(history, 40) == 25.5
    assert moving_average(history, 41) == 25.5


def test_mase_scale_matches_a_hand_computed_fixture():
    # Week 0 has no predecessor, so it adds no error. Day 8 is missing (a
    # dropped day), so day 15 falls back to t - 14 = day 1 (20).
    # Errors: day 7 |12-10| = 2, day 9 |33-30| = 3, day 10 |41-40| = 1,
    # day 11 |48-50| = 2, day 12 |63-60| = 3, day 13 |70-70| = 0,
    # day 14 |13-12| = 1, day 15 |28-20| = 8: 20 over 8 days = 2.5.
    history = {
        0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: 60.0, 6: 70.0,
        7: 12.0, 9: 33.0, 10: 41.0, 11: 48.0, 12: 63.0, 13: 70.0,
        14: 13.0, 15: 28.0,
    }
    # A scale that skipped the fallback day would give 12 / 7; one that
    # counted week 0 as zero error, 20 / 15.
    assert in_sample_naive_mae(history) == 2.5
    # Too short a history (no same-weekday predecessor): NaN, not 0.
    assert math.isnan(in_sample_naive_mae({0: 5.0, 1: 6.0}))


def test_oracle_point_is_the_poisson_median():
    means = {5: 10.3, 6: 0.0}
    predictions = forecast_days({}, [5, 6], "oracle", means)
    assert predictions == [(10.0, 10.3), (0.0, 0.0)]


# --- the oracle: parity with the simulator, calibration ------------------


def test_structural_zero_rule_closed_monday_is_not_zero_for_the_total():
    config = SimConfig(n_weeks=2)
    monday = 0  # start_day is a Monday and Monday is closed by default
    assert true_daily_mean(config, monday, "in_store") == 0.0
    assert true_daily_mean(config, monday, "total") == 90.0
    assert true_daily_mean(config, 5, "total") == 280.0 + 90.0


def parity_z(scale_factor):
    """z of (simulated total - sum of true means * scale) / sqrt(sum means)."""
    config = SimConfig(n_weeks=4, drift_per_week=0.02)
    n_days = config.n_weeks * 7
    observed_total = 0.0
    expected_total = 0.0
    for seed in TRIAL_SEEDS:
        sim = simulate(config, seed)
        observed_total += float(sim.truth_daily["total"].sum())
        day_values = range(n_days)
        for day in day_values:
            expected_total += true_daily_mean(config, day)
    return (observed_total - expected_total * scale_factor) / math.sqrt(expected_total)


def test_oracle_parity_simulated_means_match_true_daily_mean():
    assert abs(parity_z(1.0)) < 4.0


def test_control_parity_rejects_a_mean_off_by_two_percent():
    assert abs(parity_z(1.02)) > 4.0


def oracle_coverage_z(level, interval_factor):
    """(hits - expected) / Poisson-binomial SE on simulated daily totals.

    Hits use an interval built at true mean * interval_factor; the expected
    coverage and SE always assume the honest oracle interval (factor 1), so
    a biased interval shows up as a large |z|.
    """
    config = SimConfig(n_weeks=4, drift_per_week=0.01)
    n_days = config.n_weeks * 7
    hits = 0
    expected = 0.0
    variance = 0.0
    for seed in TRIAL_SEEDS:
        sim = simulate(config, seed)
        totals = list(sim.truth_daily["total"])
        day_values = range(n_days)
        for day in day_values:
            true_mean = true_daily_mean(config, day)
            used_low, used_high = poisson_interval(true_mean * interval_factor, level)
            if used_low <= totals[day] <= used_high:
                hits += 1
            low, high = poisson_interval(true_mean, level)
            probability = poisson_mass_between(true_mean, low, high)
            expected += probability
            variance += probability * (1.0 - probability)
    return (hits - expected) / math.sqrt(variance)


def test_oracle_coverage_within_three_poisson_binomial_se():
    assert abs(oracle_coverage_z(0.80, 1.0)) <= 3.0
    assert abs(oracle_coverage_z(0.95, 1.0)) <= 3.0


def test_control_coverage_check_catches_a_biased_oracle():
    # An interval centred 15% too high must miss far more than predicted.
    assert oracle_coverage_z(0.80, 1.15) < -3.0
