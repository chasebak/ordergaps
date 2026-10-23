from ordergaps.classify import CLOSED, FULL, SPREAD
from ordergaps.estimator import estimate
from ordergaps.evaluation.experiments import run_one
from ordergaps.simulator import SimConfig, simulate


def test_end_to_end_weekly_accuracy_at_healthy_share():
    config = SimConfig(n_weeks=6, vendor_share=0.25, void_rate=0.0, reset_week=None)
    metrics = run_one(config, seed=42)
    assert metrics.n_weeks_used >= 4
    assert metrics.weekly_err_median < 0.01
    # The channel split is model-based: online orders placed during open
    # hours are attributed in-store, so this error is structurally larger
    # than the counted weekly-total error above.
    assert metrics.channel_err_median < 0.25


def test_day_labels_cover_the_range():
    config = SimConfig(n_weeks=6, vendor_share=0.25, void_rate=0.0, reset_week=None)
    sim = simulate(config, seed=42)
    result = estimate(sim.observed, config.store_hours)
    labels = list(result.day_labels.values())
    assert len(labels) == len(result.daily)
    assert labels.count(FULL) > 10
    # Mondays are closed in the default schedule.
    monday_labels = []
    label_items = result.day_labels.items()
    for day, label in label_items:
        if day.weekday() == 0:
            monday_labels.append(label)
    assert monday_labels and set(monday_labels) == {CLOSED}


def test_low_share_produces_spread_days():
    config = SimConfig(n_weeks=6, vendor_share=0.01, void_rate=0.0, reset_week=None)
    sim = simulate(config, seed=42)
    result = estimate(sim.observed, config.store_hours)
    labels = list(result.day_labels.values())
    assert labels.count(SPREAD) >= 1


def test_vendor_share_of_in_store_is_recovered():
    config = SimConfig(n_weeks=6, vendor_share=0.25, void_rate=0.0, reset_week=None)
    sim = simulate(config, seed=42)
    result = estimate(sim.observed, config.store_hours)
    clean = result.weekly[result.weekly["fully_covered"]]
    shares = list(clean["vendor_share_in_store"])
    for share in shares:
        assert abs(share - 0.25) < 0.07


def test_full_day_averages_table_shape():
    config = SimConfig(n_weeks=6, vendor_share=0.25, void_rate=0.0, reset_week=None)
    sim = simulate(config, seed=42)
    result = estimate(sim.observed, config.store_hours)
    assert len(result.full_day_averages) == 7
    # A clean config (no drift, no voids, no reset) over simulated Poisson
    # arrivals must come back consistent with Poisson, not merely land on
    # SOME verdict.
    assert result.dispersion.verdict == "poisson_ok"
    assert 0.5 < result.dispersion.dispersion_index < 1.6
