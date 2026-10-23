"""Dispersion check: Poisson vs negative binomial on full-day counts.

The allocation model assumes within-session homogeneous-Poisson arrivals.
This check tests the Poisson part on the most trustworthy data available:
in-store counts of FULL days, grouped by weekday. It reports a pooled
conditional chi-square dispersion test and a method-of-moments negative
binomial (Var = mu + alpha * mu^2) log-likelihood comparison.
"""

import math
from dataclasses import dataclass

_MAX_ITERATIONS = 500
_EPSILON = 1e-12


def _gamma_p_series(a, x):
    term = 1.0 / a
    total = term
    denominator = a
    iteration_values = range(_MAX_ITERATIONS)
    for _ in iteration_values:
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) < abs(total) * _EPSILON:
            break
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    return total * math.exp(log_prefix)


def _gamma_q_continued_fraction(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    iteration_values = range(1, _MAX_ITERATIONS)
    for i in iteration_values:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    return math.exp(log_prefix) * h


def gamma_q(a, x):
    """Regularized upper incomplete gamma Q(a, x)."""
    if a <= 0.0 or x < 0.0:
        raise ValueError("gamma_q needs a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_continued_fraction(a, x)


def chi2_sf(x, dof):
    """Chi-square survival function P(X > x) with dof degrees of freedom."""
    if x <= 0.0:
        return 1.0
    return gamma_q(dof / 2.0, x / 2.0)


@dataclass
class DispersionResult:
    n_days: int
    n_groups: int
    statistic: float
    dof: int
    p_value: float
    dispersion_index: float
    alpha_mom: float
    poisson_loglik: float
    nb_loglik: float
    verdict: str  # "poisson_ok" | "overdispersed" | "underdispersed" | "insufficient_data"

    def describe(self):
        if self.verdict == "insufficient_data":
            return "dispersion check: insufficient full days to test"
        text = (
            f"dispersion check on {self.n_days} full days: index "
            f"{self.dispersion_index:.2f} (chi2 {self.statistic:.1f} on {self.dof} df, "
            f"p={self.p_value:.3f});"
        )
        if self.verdict == "overdispersed":
            text += (
                f" overdispersed vs Poisson; negative binomial preferred "
                f"(alpha={self.alpha_mom:.4f}, dlogL={self.nb_loglik - self.poisson_loglik:+.1f})"
            )
        elif self.verdict == "underdispersed":
            text += " underdispersed relative to Poisson"
        else:
            text += " consistent with within-session Poisson arrivals"
        return text


def _poisson_loglik(count, mu):
    return count * math.log(mu) - mu - math.lgamma(count + 1.0)


def _nb_loglik(count, mu, alpha):
    r = 1.0 / alpha
    log_r_part = r * math.log(r / (r + mu))
    log_mu_part = count * math.log(mu / (r + mu))
    log_gamma_part = math.lgamma(count + r) - math.lgamma(r) - math.lgamma(count + 1.0)
    return log_gamma_part + log_r_part + log_mu_part


def dispersion_check(counts_by_weekday, min_group_size=3, min_dof=5):
    """Pooled conditional dispersion test across weekday groups.

    counts_by_weekday: weekday -> list of full-day in-store counts.
    Groups with fewer than min_group_size days or zero mean are skipped.
    """
    groups = []
    weekday_items = sorted(counts_by_weekday.items())
    for weekday, raw_counts in weekday_items:
        counts = []
        for value in raw_counts:
            counts.append(float(round(value)))
        if len(counts) < min_group_size:
            continue
        mean = sum(counts) / len(counts)
        if mean <= 0.0:
            continue
        groups.append((weekday, counts, mean))

    n_days = 0
    statistic = 0.0
    dof = 0
    sq_excess_sum = 0.0  # sum of (x - mu)^2 - mu
    mu_sq_sum = 0.0
    for weekday, counts, mean in groups:
        for count in counts:
            deviation = count - mean
            statistic += deviation * deviation / mean
            sq_excess_sum += deviation * deviation - mean
            mu_sq_sum += mean * mean
            n_days += 1
        dof += len(counts) - 1

    if dof < min_dof:
        return DispersionResult(
            n_days=n_days,
            n_groups=len(groups),
            statistic=float("nan"),
            dof=dof,
            p_value=float("nan"),
            dispersion_index=float("nan"),
            alpha_mom=float("nan"),
            poisson_loglik=float("nan"),
            nb_loglik=float("nan"),
            verdict="insufficient_data",
        )

    p_value = chi2_sf(statistic, dof)
    dispersion_index = statistic / dof
    alpha_mom = 0.0
    if mu_sq_sum > 0.0 and sq_excess_sum > 0.0:
        alpha_mom = sq_excess_sum / mu_sq_sum

    poisson_loglik = 0.0
    nb_loglik = 0.0
    for weekday, counts, mean in groups:
        for count in counts:
            poisson_loglik += _poisson_loglik(count, mean)
            if alpha_mom > 0.0:
                nb_loglik += _nb_loglik(count, mean, alpha_mom)
            else:
                nb_loglik += _poisson_loglik(count, mean)

    if p_value < 0.05 and dispersion_index > 1.0 and nb_loglik > poisson_loglik:
        verdict = "overdispersed"
    elif p_value < 0.05 and dispersion_index < 1.0:
        verdict = "underdispersed"
    else:
        verdict = "poisson_ok"
    return DispersionResult(
        n_days=n_days,
        n_groups=len(groups),
        statistic=statistic,
        dof=dof,
        p_value=p_value,
        dispersion_index=dispersion_index,
        alpha_mom=alpha_mom,
        poisson_loglik=poisson_loglik,
        nb_loglik=nb_loglik,
        verdict=verdict,
    )
