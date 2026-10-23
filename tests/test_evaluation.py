import numpy as np

from ordergaps.evaluation.experiments import (
    count_inversions,
    rank_with_ties,
    run_one,
    spearman,
)
from ordergaps.simulator import SimConfig


def test_run_one_is_reproducible_with_pinned_seed():
    config = SimConfig(n_weeks=3, vendor_share=0.1, void_rate=0.02, drift_per_week=0.01)
    first = run_one(config, seed=99)
    second = run_one(config, seed=99)
    np.testing.assert_equal(first.__dict__, second.__dict__)


def test_spearman_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    decreasing = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    increasing = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, decreasing) == -1.0
    assert spearman(x, increasing) == 1.0


def test_rank_with_ties_averages():
    ranks = rank_with_ties([10.0, 20.0, 20.0, 30.0])
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_count_inversions():
    assert count_inversions([5.0, 4.0, 3.0, 2.0]) == 0
    assert count_inversions([5.0, 4.0, 4.5, 2.0]) == 1
    assert count_inversions([1.0, 2.0, 3.0]) == 2
