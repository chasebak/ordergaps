"""The uniform-in-time baseline allocator and the full-day yield count.

Why a baseline: the estimator's weekly totals are mostly ID arithmetic, so
"0.11% weekly error" means little until it sits next to the simplest rival
that gets the SAME counts. The uniform-in-time allocator takes the same gaps
(same unseen counts, same skipped anomalies) and spreads each gap's unseen
orders evenly over its wall-clock span, ignoring store hours, sessions and
learned paces. Whatever it scores is printed beside the model, including
where it wins (DECISIONS.md, "Uniform-in-time baseline: print it, losses
included").

Why the two can only differ at week boundaries: both allocators hand out
every unseen order of every gap exactly once, so the total of any week is
seen orders + the unseen orders of the gaps inside it + some share of the
gaps that straddle its start or end. A week that no gap with unseen orders
straddles gets identical totals from both (tests/test_baseline.py).
"""

from datetime import timedelta

import pandas as pd

from ordergaps.attribution import split_span
from ordergaps.classify import CLOSED, FULL

# Two weekly totals closer than this (in orders) count as the same total:
# float sums in a different order differ by ~1e-12, a real allocation
# difference by far more than this.
ALLOCATOR_TIE_ORDERS = 1e-6


def uniform_unseen_by_day(gaps):
    """{day: unseen orders} with each gap spread evenly over wall-clock time.

    Rule: a gap (t1, t2) with u unseen orders gives day d the amount
    u * minutes of (t1, t2) on d / minutes of (t1, t2). A zero-length gap
    puts all u on the day of t1 (the model does the same).
    Failure mode: if this read different gaps than the model (or dropped a
    gap), the baseline would be scored on different counts; the
    conservation test requires both allocators to hand out the same total.
    """
    unseen_by_day = {}
    no_sessions = {}
    for gap in gaps:
        unseen = gap.unseen
        if unseen <= 0:
            continue
        spans = split_span(gap.t1, gap.t2, no_sessions)
        if not spans:
            day = gap.t1.date()
            unseen_by_day[day] = unseen_by_day.get(day, 0.0) + float(unseen)
            continue
        span_minutes = []
        total_minutes = 0.0
        for span in spans:
            minutes = span.open_minutes + span.closed_minutes
            span_minutes.append(minutes)
            total_minutes += minutes
        pairs = zip(spans, span_minutes)
        for span, minutes in pairs:
            amount = float(unseen) * minutes / total_minutes
            unseen_by_day[span.day] = unseen_by_day.get(span.day, 0.0) + amount
    return unseen_by_day


def uniform_weekly_totals(result):
    """{(iso_year, iso_week): total} under the uniform allocator.

    Rule: per day, seen orders (register + online) from the estimator's
    daily table + uniform_unseen_by_day(result.gaps); summed per ISO week,
    over exactly the days the estimator's own weekly table sums.
    """
    unseen_by_day = uniform_unseen_by_day(result.gaps)
    totals = {}
    daily_tuples = list(result.daily.itertuples(index=False))
    for row in daily_tuples:
        iso = row.day.isocalendar()
        key = (iso.year, iso.week)
        seen = row.seen_register + row.seen_online
        day_total = seen + unseen_by_day.get(row.day, 0.0)
        totals[key] = totals.get(key, 0.0) + day_total
    return totals


def model_weekly_totals(result):
    """{(iso_year, iso_week): est_total} from the estimator's weekly table."""
    totals = {}
    weekly_tuples = list(result.weekly.itertuples(index=False))
    for row in weekly_tuples:
        totals[(row.iso_year, row.iso_week)] = float(row.est_total)
    return totals


def week_start_of(iso_year, iso_week):
    """Monday 00:00 of an ISO week, as a Timestamp."""
    monday = pd.Timestamp.fromisocalendar(iso_year, iso_week, 1)
    return monday


def gap_straddles(gap, boundary):
    """True when a gap with unseen orders spans the instant `boundary`.

    The gap's span is the open interval (t1, t2), so a seen order exactly at
    the boundary means no gap straddles it. A gap with no unseen orders hands
    out nothing, so it cannot make the allocators differ.
    """
    if gap.unseen <= 0:
        return False
    return gap.t1 < boundary < gap.t2


def straddled_weeks(gaps, week_keys):
    """The ISO weeks (of week_keys) whose start or end a gap straddles."""
    one_week = timedelta(days=7)
    straddled = set()
    for key in week_keys:
        iso_year, iso_week = key
        start = week_start_of(iso_year, iso_week)
        end = start + one_week
        for gap in gaps:
            if gap_straddles(gap, start) or gap_straddles(gap, end):
                straddled.add(key)
                break
    return straddled


def full_day_yield_counts(day_labels):
    """(full days, non-closed days) over the estimator's day labels.

    Rule: yield = FULL ÷ (every label except CLOSED), on every labelled day
    from the first to the last seen order. It is a count of the estimator's
    own labels, not a comparison with truth; whether a FULL label deserves
    trust is measured separately (README section 3: per-day error on full
    days). Failure mode: a looser pin rule raises the yield without making
    full days any more exact, which is why it is printed next to that error.
    """
    n_full = 0
    n_open = 0
    labels = day_labels.values()
    for label in labels:
        if label == CLOSED:
            continue
        n_open += 1
        if label == FULL:
            n_full += 1
    return n_full, n_open
