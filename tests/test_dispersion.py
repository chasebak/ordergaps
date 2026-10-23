import numpy as np

from ordergaps.dispersion import chi2_sf, dispersion_check


def counts_by_weekday_from_matrix(matrix):
    counts = {}
    weekday = 1
    for row in matrix:
        counts[weekday] = list(row)
        weekday += 1
    return counts


def test_chi2_sf_known_values():
    assert chi2_sf(0.0, 5) == 1.0
    assert abs(chi2_sf(3.841, 1) - 0.05) < 0.001
    assert abs(chi2_sf(31.410, 20) - 0.05) < 0.001
    assert chi2_sf(1000.0, 5) < 1e-10


def test_poisson_counts_pass_the_check():
    rng = np.random.default_rng(123)
    matrix = rng.poisson(150.0, size=(5, 20))
    result = dispersion_check(counts_by_weekday_from_matrix(matrix))
    assert result.verdict == "poisson_ok"
    assert 0.5 < result.dispersion_index < 1.6


def test_overdispersed_counts_prefer_negative_binomial():
    rng = np.random.default_rng(123)
    # Negative binomial with Var = mu + alpha * mu^2, alpha = 0.05, mu = 150.
    alpha = 0.05
    mu = 150.0
    r = 1.0 / alpha
    p = r / (r + mu)
    matrix = rng.negative_binomial(r, p, size=(5, 20))
    result = dispersion_check(counts_by_weekday_from_matrix(matrix))
    assert result.verdict == "overdispersed"
    assert result.p_value < 0.01
    assert result.nb_loglik > result.poisson_loglik
    assert result.alpha_mom > 0.01


def test_insufficient_data_is_reported_not_guessed():
    result = dispersion_check({1: [5.0, 6.0]})
    assert result.verdict == "insufficient_data"
    assert "insufficient" in result.describe()
