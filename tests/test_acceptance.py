"""The frozen acceptance band for the hero experiment.

The simulator regime was tuned on trial seeds; the seeds (run_all.FROZEN_SEEDS) were frozen
afterwards and the committed results use exactly this configuration. The
acceptance thresholds (Spearman rho <= -0.8, at most one inversion) were
set before the frozen seeds were generated.
"""

from ordergaps.evaluation.experiments import acceptance_check, error_curve


def test_frozen_acceptance_band(frozen_records):
    # frozen_records (tests/conftest.py) = run_matrix(frozen_base_config(), SHARES, FROZEN_SEEDS)
    records = frozen_records
    curve = error_curve(records)
    check = acceptance_check(curve)
    assert check["spearman"] <= -0.8
    assert check["inversions"] <= 1
    assert check["passed"]
    # Exactness of the ID arithmetic holds across the whole matrix.
    assert int(records["n_windows_exact"].sum()) == int(records["n_windows"].sum())
