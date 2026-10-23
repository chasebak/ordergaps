import pandas as pd
import pytest

from ordergaps.estimator import estimate
from ordergaps.evaluation.experiments import run_one
from ordergaps.gaps import Gap
from ordergaps.simulator import SimConfig, simulate


def clean_config():
    return SimConfig(n_weeks=3, vendor_share=0.2, void_rate=0.0, reset_week=None)


def test_unseen_is_exact_id_arithmetic():
    gap = Gap(100, pd.Timestamp("2025-03-04 12:00"), 108, pd.Timestamp("2025-03-04 13:00"))
    assert gap.unseen == 7


def test_window_totals_match_ground_truth_exactly():
    metrics = run_one(clean_config(), seed=11)
    assert metrics.n_windows > 100
    assert metrics.n_windows_exact == metrics.n_windows


def test_estimates_conserve_seen_plus_attributed():
    sim = simulate(clean_config(), seed=11)
    result = estimate(sim.observed, sim.config.store_hours)
    estimated_total = float(result.daily["est_total"].sum())
    expected_total = len(sim.observed) + result.unseen_attributed
    assert estimated_total == pytest.approx(expected_total)
    assert result.unseen_skipped == 0
