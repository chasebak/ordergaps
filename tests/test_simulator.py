import pandas as pd

from ordergaps.simulator import SimConfig, simulate, with_share


def test_same_seed_reproduces_exactly():
    config = SimConfig(n_weeks=2, vendor_share=0.1)
    first = simulate(config, seed=5)
    second = simulate(config, seed=5)
    pd.testing.assert_frame_equal(first.orders, second.orders)
    pd.testing.assert_frame_equal(first.observed, second.observed)


def test_numbers_are_sequential_without_reset():
    config = SimConfig(n_weeks=2, vendor_share=0.1, reset_week=None)
    sim = simulate(config, seed=5)
    differences = sim.orders["order_number"].diff().dropna()
    assert (differences == 1).all()


def test_reset_restarts_numbering():
    config = SimConfig(n_weeks=4, vendor_share=0.1, reset_week=2, reset_new_base=101)
    sim = simulate(config, seed=5)
    after_reset = sim.orders[sim.orders["created_at"] >= sim.reset_time]
    assert int(after_reset["order_number"].min()) == 101
    differences = after_reset["order_number"].diff().dropna()
    assert (differences == 1).all()


def test_closed_day_has_no_in_store_orders():
    config = SimConfig(n_weeks=3, vendor_share=0.1)
    sim = simulate(config, seed=5)
    daily_tuples = list(sim.truth_daily.itertuples(index=False))
    for row in daily_tuples:
        if row.day.weekday() == 0:  # Monday closed in the default schedule
            assert row.in_store == 0


def test_vendor_sets_are_nested_across_shares():
    config = SimConfig(n_weeks=2, vendor_share=0.1)
    small = simulate(config, seed=5)
    large = simulate(with_share(config, 0.3), seed=5)
    small_numbers = set(small.observed["order_number"])
    large_numbers = set(large.observed["order_number"])
    assert small_numbers <= large_numbers


def test_fraction_observed_tracks_share_and_voids():
    config = SimConfig(n_weeks=4, vendor_share=0.2, void_rate=0.02)
    sim = simulate(config, seed=5)
    assert 0.15 < sim.fraction_observed < 0.25
