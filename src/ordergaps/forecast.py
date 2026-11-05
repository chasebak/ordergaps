"""The easiest daily-order forecasts, their Poisson intervals, and the oracle.

A history is a dict {day_index: value} of TRAINING days only (days may be
missing: the backtest drops the incomplete last day and reset spans). Days
with the same day_index mod 7 share a weekday. Every forecaster reads the
history and nothing else, so it cannot see the future by construction; the
backtest decides what the history contains.

Models, easiest first:
- seasonal_naive: "same weekday last week", with a walk-back fallback;
- moving_average: the last 28 training values, weekday-blind on purpose;
- weekday_average: the last 4 training values on the target's weekday;
- the oracle (oracle_mean): the simulator's TRUE expected count, the floor.

Intervals are central Poisson intervals around the point forecast. The
quantiles come from one cumulative-pmf loop per call (log-space pmf with
math.lgamma, so large means do not underflow at k = 0).
"""

import math

WEEK_DAYS = 7
NAIVE_MAX_WEEKS_BACK = 52
MOVING_AVERAGE_DAYS = 28
WEEKDAY_AVERAGE_WEEKS = 4
NAN = float("nan")


def poisson_quantiles(mean, probabilities):
    """Smallest k with Poisson(mean) CDF(k) >= p, for each p, in one loop.

    Rule: walk k = 0, 1, 2, ... accumulating pmf(k) = exp(-mean + k ln mean
    - lgamma(k + 1)); each requested p is answered the first time the
    cumulative sum reaches it. mean <= 0 is a point mass at 0.
    Example (a test fixture): mean 10 gives 4 at p = 0.025 and 17 at 0.975.
    Failure mode: p = 1.0 never terminates on a true Poisson, so p must be
    < 1; floating-point CDF never quite reaching p near 1 is guarded by a
    cap far in the tail (mean + 50 sd + 50).
    """
    answers = {}
    for probability in probabilities:
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"probability must be in [0, 1), got {probability}")
        answers[probability] = None
    if mean <= 0.0:
        results = []
        for probability in probabilities:
            results.append(0)
        return results
    pending = sorted(answers)
    pending_index = 0
    n_pending = len(pending)
    log_mean = math.log(mean)
    cap = int(mean + 50.0 * math.sqrt(mean) + 50.0)
    cumulative = 0.0
    k = 0
    while pending_index < n_pending and k <= cap:
        log_pmf = -mean + k * log_mean - math.lgamma(k + 1.0)
        cumulative += math.exp(log_pmf)
        while pending_index < n_pending and cumulative >= pending[pending_index]:
            answers[pending[pending_index]] = k
            pending_index += 1
        k += 1
    while pending_index < n_pending:
        answers[pending[pending_index]] = cap
        pending_index += 1
    results = []
    for probability in probabilities:
        results.append(answers[probability])
    return results


def poisson_quantile(mean, probability):
    """One Poisson quantile (see poisson_quantiles for the rule)."""
    values = poisson_quantiles(mean, [probability])
    return values[0]


def poisson_median(mean):
    """The Poisson median: the MAE-optimal point forecast for a Poisson count."""
    return poisson_quantile(mean, 0.5)


def poisson_interval(mean, level):
    """Central Poisson interval [q((1 - level)/2), q((1 + level)/2)] around mean.

    Whole-number counts make it over-cover: at mean 10 the 80% interval
    holds about 0.849 of the mass. The oracle's expected coverage, not the
    nominal level, is what a calibrated interval should hit.
    """
    tail = (1.0 - level) / 2.0
    bounds = poisson_quantiles(mean, [tail, 1.0 - tail])
    return bounds[0], bounds[1]


def poisson_mass_between(mean, low, high):
    """P(low <= Y <= high) for Y ~ Poisson(mean), by one cumulative loop.

    Used for the achievable coverage: the exact chance that a day's count
    lands inside the oracle's interval.
    """
    if high < low:
        return 0.0
    if mean <= 0.0:
        if low <= 0 <= high:
            return 1.0
        return 0.0
    log_mean = math.log(mean)
    total = 0.0
    k = max(0, int(low))
    last = int(high)
    while k <= last:
        log_pmf = -mean + k * log_mean - math.lgamma(k + 1.0)
        total += math.exp(log_pmf)
        k += 1
    return min(total, 1.0)


def _earliest_day(history):
    """The first training day, or None for an empty history."""
    days = sorted(history)
    if not days:
        return None
    return days[0]


def seasonal_naive(history, target_day):
    """Same weekday last week: history[t - 7], else t - 14, and so on back.

    Rule: the most recent training day with the target's weekday. The
    walk-back matters right after an origin, where the last open day is
    dropped (incomplete) and its weekday falls back to t - 14.
    Failure mode: returns NaN when no same-weekday day exists in the history.
    """
    earliest = _earliest_day(history)
    if earliest is None:
        return NAN
    lag = WEEK_DAYS
    max_lag = WEEK_DAYS * NAIVE_MAX_WEEKS_BACK
    while lag <= max_lag:
        candidate = target_day - lag
        if candidate < earliest:
            break
        if candidate in history:
            return float(history[candidate])
        lag += WEEK_DAYS
    return NAN


def moving_average(history, target_day, window_days=MOVING_AVERAGE_DAYS):
    """Mean of the last window_days training values before target_day.

    Weekday-blind on purpose: it is the deliberately weak baseline that
    shows what ignoring a 90-to-370-orders-a-day weekday profile costs.
    """
    days = sorted(history)
    usable = []
    for day in days:
        if day < target_day:
            usable.append(day)
    recent = usable[-window_days:]
    if not recent:
        return NAN
    total = 0.0
    for day in recent:
        total += float(history[day])
    return total / len(recent)


def weekday_average(history, target_day, n_values=WEEKDAY_AVERAGE_WEEKS):
    """Mean of the last n_values training values on the target's weekday.

    Theory (no drift): error variance mu + mu / 4 = 1.25 mu vs naive's 2 mu.
    Failure mode: under strong drift a 4-week mean lags the level by about
    2.5 weeks of growth.
    """
    earliest = _earliest_day(history)
    if earliest is None:
        return NAN
    values = []
    n_found = 0
    lag = WEEK_DAYS
    while n_found < n_values:
        candidate = target_day - lag
        if candidate < earliest:
            break
        if candidate in history:
            values.append(float(history[candidate]))
            n_found += 1
        lag += WEEK_DAYS
    if not values:
        return NAN
    return sum(values) / len(values)


def in_sample_naive_mae(history):
    """MASE scale: MAE of seasonal naive predicting each training day from
    the earlier training days (t - 7, else t - 14, ...).

    Days with no earlier same-weekday value are skipped. Failure mode: NaN
    when no training day has a predecessor (too short a history).
    """
    days = sorted(history)
    errors = []
    for day in days:
        # seasonal_naive only reads days strictly before its target.
        prediction = seasonal_naive(history, day)
        if math.isnan(prediction):
            continue
        errors.append(abs(float(history[day]) - prediction))
    if not errors:
        return NAN
    return sum(errors) / len(errors)


POINT_FORECASTERS = {
    "naive": seasonal_naive,
    "avg28": moving_average,
    "weekday_avg": weekday_average,
}
