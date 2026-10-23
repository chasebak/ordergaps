import pandas as pd

from ordergaps.estimator import estimate
from ordergaps.evaluation.experiments import run_one
from ordergaps.simulator import SimConfig, simulate


def observed_frame(rows):
    numbers = []
    times = []
    channels = []
    for number, time_text, channel in rows:
        numbers.append(number)
        times.append(pd.Timestamp(time_text))
        channels.append(channel)
    return pd.DataFrame(
        {"order_number": numbers, "created_at": times, "channel": channels}
    )


def anomaly_kinds(result):
    kinds = []
    for anomaly in result.anomalies:
        kinds.append(anomaly["kind"])
    return kinds


def test_out_of_order_timestamp_skips_gap():
    observed = observed_frame(
        [
            (1000, "2025-03-04 12:00", "register"),
            (1010, "2025-03-04 11:30", "register"),  # timestamp regressed
            (1025, "2025-03-04 13:00", "register"),
        ]
    )
    result = estimate(observed)
    kinds = anomaly_kinds(result)
    assert "out_of_order_timestamp" in kinds
    assert result.unseen_skipped == 9  # orders 1001..1009 dropped, not guessed
    assert len(result.gaps) == 1  # only 1010 -> 1025 survives


def test_duplicate_order_number_is_an_anomaly_not_a_crash():
    observed = observed_frame(
        [
            (1000, "2025-03-04 12:00", "register"),
            (1000, "2025-03-04 12:01", "register"),
            (1005, "2025-03-04 13:00", "register"),
        ]
    )
    result = estimate(observed)
    kinds = anomaly_kinds(result)
    assert "duplicate_number" in kinds


def test_numbering_reset_is_detected_and_split():
    config = SimConfig(n_weeks=4, vendor_share=0.3, void_rate=0.0, reset_week=2)
    sim = simulate(config, seed=21)
    result = estimate(sim.observed, config.store_hours)
    kinds = anomaly_kinds(result)
    assert kinds.count("numbering_reset") == 1

    # The destroyed cross-reset span runs from the previous era's last
    # observation to the first post-reset one; the anomaly records both ends.
    reset_anomaly = None
    for anomaly in result.anomalies:
        if anomaly["kind"] == "numbering_reset":
            reset_anomaly = anomaly
    before_reset = sim.observed[sim.observed["created_at"] < sim.reset_time]
    after_reset = sim.observed[sim.observed["created_at"] >= sim.reset_time]
    assert reset_anomaly["span_start"] == before_reset["created_at"].max()
    assert reset_anomaly["at"] == after_reset["created_at"].min()

    # EVERY ISO week overlapping the span is flagged. Here the span starts
    # late Sunday of ISO week 11 and ends early Monday of ISO week 12, so
    # both weeks flag -- the week BEFORE the reset holds most of the span
    # and must not be scored as clean.
    flagged = result.weekly[result.weekly["has_anomaly"]]
    flagged_weeks = set()
    flagged_tuples = list(flagged.itertuples(index=False))
    for row in flagged_tuples:
        flagged_weeks.add((row.iso_year, row.iso_week))
    assert flagged_weeks == {(2025, 11), (2025, 12)}

    # Gap arithmetic still exact within each era.
    metrics = run_one(config, seed=21)
    assert metrics.n_windows_exact == metrics.n_windows


def test_void_orders_consume_numbers_and_are_counted_as_created():
    # 15% of orders are void; truth totals INCLUDE them. If the estimator
    # measured fulfilled orders instead of created ones, weekly error would
    # sit near 15%, not under 2%.
    config = SimConfig(n_weeks=4, vendor_share=0.4, void_rate=0.15, reset_week=None)
    metrics = run_one(config, seed=31)
    assert metrics.n_weeks_used >= 2
    assert metrics.weekly_err_median < 0.02
