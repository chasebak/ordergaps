"""The uniform-in-time baseline and the full-day yield.

Logic check: a week that no gap straddles has identical totals under both
allocators. Both hand out
every unseen order of every gap exactly once, so a week's total can only
differ through a gap that crosses its boundary. The control (the same
simulation without the boundary-pinning orders) must show straddled weeks
whose totals DO differ, or the check proves nothing.

Trial seeds only (never the frozen evaluation seeds).
"""

import math

import pandas as pd
import pytest

from ordergaps.baseline import (
    ALLOCATOR_TIE_ORDERS,
    full_day_yield_counts,
    gap_straddles,
    model_weekly_totals,
    straddled_weeks,
    uniform_unseen_by_day,
    uniform_weekly_totals,
)
from ordergaps.classify import CLOSED, FULL, PARTIAL, SPREAD
from ordergaps.estimator import estimate
from ordergaps.evaluation.experiments import baseline_and_yield, versus_uniform
from ordergaps.gaps import Gap
from ordergaps.simulator import SimConfig, simulate

TRIAL_SEED = 7
LOW_SHARE = 0.02


def ts(text):
    return pd.Timestamp(text)


def test_uniform_spreads_a_gap_by_clock_minutes():
    # Sunday 18:00 -> Tuesday 00:00: 6 h on Sunday, 24 h on Monday, 10 unseen.
    gap = Gap(100, ts("2025-03-09 18:00"), 111, ts("2025-03-11 00:00"))
    by_day = uniform_unseen_by_day([gap])
    assert by_day[ts("2025-03-09").date()] == pytest.approx(2.0)
    assert by_day[ts("2025-03-10").date()] == pytest.approx(8.0)
    assert sorted(by_day) == [ts("2025-03-09").date(), ts("2025-03-10").date()]


def test_uniform_zero_length_and_empty_gaps():
    same_instant = Gap(1, ts("2025-03-05 12:00"), 5, ts("2025-03-05 12:00"))
    no_unseen = Gap(5, ts("2025-03-05 12:00"), 6, ts("2025-03-06 12:00"))
    by_day = uniform_unseen_by_day([same_instant, no_unseen])
    assert by_day == {ts("2025-03-05").date(): 3.0}


def test_gap_straddles_is_strict_and_ignores_empty_gaps():
    boundary = ts("2025-03-10 00:00")
    crossing = Gap(1, ts("2025-03-09 23:00"), 4, ts("2025-03-10 01:00"))
    ends_on_boundary = Gap(1, ts("2025-03-09 23:00"), 4, boundary)
    starts_on_boundary = Gap(1, boundary, 4, ts("2025-03-10 01:00"))
    crossing_but_empty = Gap(1, ts("2025-03-09 23:00"), 2, ts("2025-03-10 01:00"))
    assert gap_straddles(crossing, boundary)
    assert not gap_straddles(ends_on_boundary, boundary)
    assert not gap_straddles(starts_on_boundary, boundary)
    assert not gap_straddles(crossing_but_empty, boundary)


def low_share_simulation():
    config = SimConfig(n_weeks=3, vendor_share=LOW_SHARE, void_rate=0.02)
    return config, simulate(config, TRIAL_SEED)


def boundary_pinned_observed(sim):
    """The vendor's orders plus, at every interior Monday 00:00, the last
    store order before it and the first one at or after it. Those two are
    consecutive store orders, so the gap between them holds no unseen order
    and no gap with unseen orders straddles any week boundary."""
    orders = sim.orders.sort_values("created_at", kind="stable")
    start = pd.Timestamp(sim.config.start_day)
    extra_rows = []
    week_values = range(1, sim.config.n_weeks)
    for week in week_values:
        boundary = start + pd.Timedelta(days=7 * week)
        before = orders[orders["created_at"] < boundary]
        after = orders[orders["created_at"] >= boundary]
        extra_rows.append(before.iloc[-1])
        extra_rows.append(after.iloc[0])
    extra = pd.DataFrame(extra_rows)[["order_number", "created_at", "channel"]]
    extra["items"] = 1
    extra["total"] = 0.0
    combined = pd.concat([sim.observed, extra], ignore_index=True)
    combined = combined.drop_duplicates(subset="order_number")
    return combined.sort_values("created_at", kind="stable").reset_index(drop=True)


def weekly_differences(result):
    """{week: model total - uniform total} for every week of the estimate."""
    model = model_weekly_totals(result)
    uniform = uniform_weekly_totals(result)
    differences = {}
    keys = sorted(model)
    for key in keys:
        differences[key] = model[key] - uniform[key]
    return differences


def test_a_week_no_gap_straddles_has_identical_totals():
    config, sim = low_share_simulation()
    observed = boundary_pinned_observed(sim)
    result = estimate(observed, config.store_hours)
    differences = weekly_differences(result)
    straddled = straddled_weeks(result.gaps, differences.keys())
    assert len(differences) == config.n_weeks
    assert straddled == set()
    difference_items = differences.items()
    for key, difference in difference_items:
        assert abs(difference) <= ALLOCATOR_TIE_ORDERS, key


def test_control_straddled_weeks_do_differ():
    # Control that must fail the identity: without the pinning orders, low
    # share gaps cross week boundaries and reach into open hours, where the
    # model's learned pace and the clock disagree.
    config, sim = low_share_simulation()
    result = estimate(sim.observed, config.store_hours)
    differences = weekly_differences(result)
    straddled = straddled_weeks(result.gaps, differences.keys())
    assert straddled != set()
    n_differ = 0
    difference_items = differences.items()
    for key, difference in difference_items:
        if abs(difference) > ALLOCATOR_TIE_ORDERS:
            n_differ += 1
            assert key in straddled
    assert n_differ >= 1


def test_both_allocators_hand_out_the_same_unseen_total():
    config, sim = low_share_simulation()
    result = estimate(sim.observed, config.store_hours)
    uniform = uniform_unseen_by_day(result.gaps)
    uniform_total = 0.0
    amounts = uniform.values()
    for amount in amounts:
        uniform_total += amount
    assert uniform_total == pytest.approx(result.unseen_attributed, rel=1e-12)
    # Control: drop one gap with unseen orders and the totals part ways.
    first_with_unseen = 0
    while result.gaps[first_with_unseen].unseen <= 0:
        first_with_unseen += 1
    fewer_gaps = list(result.gaps)
    del fewer_gaps[first_with_unseen]
    fewer_total = 0.0
    fewer_amounts = uniform_unseen_by_day(fewer_gaps).values()
    for amount in fewer_amounts:
        fewer_total += amount
    assert fewer_total != pytest.approx(result.unseen_attributed, rel=1e-12)


def test_full_day_yield_counts_skip_closed_days():
    labels = {1: FULL, 2: FULL, 3: PARTIAL, 4: SPREAD, 5: CLOSED, 6: CLOSED}
    assert full_day_yield_counts(labels) == (2, 4)
    assert full_day_yield_counts({1: CLOSED}) == (0, 0)


def test_versus_uniform_prints_a_loss_as_a_loss():
    assert versus_uniform(0.003771, 0.005369) == "model lower"
    # Control: the baseline winning must not be hidden as a tie.
    assert versus_uniform(0.003048, 0.002563) == "uniform lower"
    # Tie only when the two print the same at 4 decimals of a percent.
    assert versus_uniform(0.0011080, 0.0011081) == "tie"
    assert versus_uniform(0.0011080, 0.0011090) == "model lower"


def test_baseline_and_yield_aggregates_like_the_error_curve():
    records = pd.DataFrame(
        {
            "share": [0.1, 0.1, 0.1],
            "weekly_err_median": [0.01, 0.03, 0.02],
            "uniform_weekly_err_median": [0.02, 0.05, 0.04],
            "n_weeks_used": [8, 8, 7],
            "n_weeks_allocators_differ": [1, 0, 2],
            "n_full_days": [3, 4, 5],
            "n_open_days": [10, 10, 10],
        }
    )
    table = baseline_and_yield(records)
    row = table.iloc[0]
    assert row["model_weekly_err"] == pytest.approx(0.02)
    assert row["uniform_weekly_err"] == pytest.approx(0.04)
    assert row["model_minus_uniform"] == pytest.approx(-0.02)
    assert row["verdict"] == "model lower"
    assert row["n_clean_weeks"] == 23
    assert row["n_weeks_allocators_differ"] == 3
    assert row["full_day_yield"] == pytest.approx(12 / 30)
    zero_open = records.assign(n_open_days=0, n_full_days=0)
    assert math.isnan(baseline_and_yield(zero_open).iloc[0]["full_day_yield"])
