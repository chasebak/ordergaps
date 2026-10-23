"""Learned in-store pace: orders per open minute, per weekday.

Only gaps that sit FULLY inside one session are used: within a session the
model attributes every unseen order to the in-store channel, so the count
per minute of such a gap estimates the store's open-hours pace directly.
Each gap contributes n2 - n1 arrivals over t2 - t1 minutes (the right
endpoint's own arrival is included so that back-to-back gaps telescope).
"""

from dataclasses import dataclass, field


@dataclass
class WeekdayRates:
    per_weekday: dict  # weekday -> learned rate (orders per open minute) or None
    pooled: float
    minutes_per_weekday: dict = field(default_factory=dict)
    counts_per_weekday: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)  # weekday -> "learned" | "pooled"

    def rate_for(self, weekday):
        rate = self.per_weekday.get(weekday)
        if rate is None:
            return self.pooled
        return rate


def gap_is_fully_inside(gap, sessions):
    day = gap.t1.date()
    if day != gap.t2.date():
        return False
    session = sessions.get(day)
    if session is None:
        return False
    return gap.t1 >= session.start and gap.t2 <= session.end


def learn_weekday_rates(gaps, sessions, params, fallback_count=0, fallback_minutes=0.0):
    """Per-weekday orders-per-open-minute from fully-inside gaps.

    Weekdays with less than params.min_rate_minutes of fully-inside gap
    time fall back to the pooled rate. When no fully-inside gap exists at
    all, the pooled rate falls back to fallback_count / fallback_minutes
    (seen register orders over total session minutes).
    """
    counts = {}
    minutes = {}
    for gap in gaps:
        if not gap_is_fully_inside(gap, sessions):
            continue
        span_minutes = (gap.t2 - gap.t1).total_seconds() / 60.0
        if span_minutes <= 0:
            continue
        weekday = gap.t1.weekday()
        counts[weekday] = counts.get(weekday, 0) + (gap.n2 - gap.n1)
        minutes[weekday] = minutes.get(weekday, 0.0) + span_minutes
    total_count = 0
    total_minutes = 0.0
    weekday_values = range(7)
    for weekday in weekday_values:
        total_count += counts.get(weekday, 0)
        total_minutes += minutes.get(weekday, 0.0)
    if total_minutes > 0:
        pooled = total_count / total_minutes
    elif fallback_minutes > 0:
        pooled = fallback_count / fallback_minutes
    else:
        pooled = 0.0
    per_weekday = {}
    source = {}
    for weekday in weekday_values:
        weekday_minutes = minutes.get(weekday, 0.0)
        if weekday_minutes >= params.min_rate_minutes:
            per_weekday[weekday] = counts.get(weekday, 0) / weekday_minutes
            source[weekday] = "learned"
        else:
            per_weekday[weekday] = None
            source[weekday] = "pooled"
    return WeekdayRates(
        per_weekday=per_weekday,
        pooled=pooled,
        minutes_per_weekday=minutes,
        counts_per_weekday=counts,
        source=source,
    )
