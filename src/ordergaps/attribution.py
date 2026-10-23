"""Temporal allocation: split a gap's wall-clock span into per-day open and
closed minutes, then attribute its unseen orders to channels and days.

Within a segment, spreading proportionally to minutes is exactly the
conditional distribution of homogeneous-Poisson arrivals given the count;
the dispersion check in dispersion.py reports how well that assumption
holds on this data.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd


@dataclass
class DaySpan:
    day: object  # datetime.date
    open_minutes: float
    closed_minutes: float


def split_span(t1, t2, sessions):
    """Per-day open/closed minutes of the open interval (t1, t2)."""
    spans = []
    if t2 <= t1:
        return spans
    one_day = timedelta(days=1)
    current_day = t1.date()
    last_day = t2.date()
    while current_day <= last_day:
        day_begin = max(t1, pd.Timestamp(current_day))
        next_midnight = pd.Timestamp(current_day) + pd.Timedelta(days=1)
        day_end = min(t2, next_midnight)
        total_minutes = (day_end - day_begin).total_seconds() / 60.0
        open_minutes = 0.0
        session = sessions.get(current_day)
        if session is not None:
            overlap_start = max(day_begin, session.start)
            overlap_end = min(day_end, session.end)
            if overlap_end > overlap_start:
                open_minutes = (overlap_end - overlap_start).total_seconds() / 60.0
        closed_minutes = total_minutes - open_minutes
        closed_minutes = max(closed_minutes, 0.0)
        if total_minutes > 0.0:
            spans.append(DaySpan(current_day, open_minutes, closed_minutes))
        current_day = current_day + one_day
    return spans


@dataclass
class GapAttribution:
    in_store_by_day: dict = field(default_factory=dict)
    online_by_day: dict = field(default_factory=dict)
    kind: str = "empty"  # empty | inside | outside | mixed

    @property
    def in_store_total(self):
        total = 0.0
        amounts = self.in_store_by_day.values()
        for amount in amounts:
            total += amount
        return total

    @property
    def online_total(self):
        total = 0.0
        amounts = self.online_by_day.values()
        for amount in amounts:
            total += amount
        return total


def attribute_gap(gap, sessions, rates):
    """Attribute a gap's unseen orders to (day, channel).

    Fully inside sessions -> all in-store. Fully outside -> all online /
    after-hours. Mixed -> expected in-store = sum over days of
    open_minutes * rate(weekday), in-store = min(unseen, expected), online
    = remainder; in-store unseen spread across days proportionally to
    minutes * rate, online proportionally to closed minutes per day.
    """
    attribution = GapAttribution()
    unseen = gap.unseen
    if unseen <= 0:
        return attribution
    spans = split_span(gap.t1, gap.t2, sessions)
    if not spans:
        # Zero-length span: put everything on the day of t1.
        day = gap.t1.date()
        session = sessions.get(day)
        inside = session is not None and session.start <= gap.t1 <= session.end
        if inside:
            attribution.in_store_by_day[day] = float(unseen)
            attribution.kind = "inside"
        else:
            attribution.online_by_day[day] = float(unseen)
            attribution.kind = "outside"
        return attribution

    total_open = 0.0
    total_closed = 0.0
    expected_total = 0.0
    expected_by_day = {}
    for span in spans:
        total_open += span.open_minutes
        total_closed += span.closed_minutes
        rate = rates.rate_for(span.day.weekday())
        expected = span.open_minutes * rate
        expected_by_day[span.day] = expected
        expected_total += expected

    if total_closed <= 0.0:
        in_store_total = float(unseen)
        online_total = 0.0
        attribution.kind = "inside"
    elif total_open <= 0.0:
        in_store_total = 0.0
        online_total = float(unseen)
        attribution.kind = "outside"
    else:
        in_store_total = min(float(unseen), expected_total)
        online_total = float(unseen) - in_store_total
        attribution.kind = "mixed"

    if in_store_total > 0.0:
        if expected_total > 0.0:
            for span in spans:
                weight = expected_by_day[span.day] / expected_total
                amount = in_store_total * weight
                if amount > 0.0:
                    attribution.in_store_by_day[span.day] = amount
        elif total_open > 0.0:
            # All learned rates are zero: fall back to open minutes alone.
            for span in spans:
                if span.open_minutes > 0.0:
                    amount = in_store_total * span.open_minutes / total_open
                    attribution.in_store_by_day[span.day] = amount
    if online_total > 0.0 and total_closed > 0.0:
        for span in spans:
            if span.closed_minutes > 0.0:
                amount = online_total * span.closed_minutes / total_closed
                attribution.online_by_day[span.day] = amount
    return attribution
