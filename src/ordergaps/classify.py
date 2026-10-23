"""Day classification: full / partial / spread / closed.

Only FULL days enter per-day averages and the dispersion check; spread
days count in weekly totals only. A session end is "pinned" when the
expected number of unseen orders between it and the nearest seen order
inside the session is small: at most max(pin_tolerance_orders,
pin_tolerance_fraction * the day's expected total), so a full day's
count is exact up to a small boundary spill.
"""

FULL = "full"
PARTIAL = "partial"
SPREAD = "spread"
CLOSED = "closed"


def classify_days(sessions, seen_times_by_day, rates, params):
    labels = {}
    session_items = sessions.items()
    for day, session in session_items:
        if session is None:
            labels[day] = CLOSED
            continue
        seen_times = seen_times_by_day.get(day, [])
        inside_times = []
        for ts in seen_times:
            if session.start <= ts <= session.end:
                inside_times.append(ts)
        if not inside_times:
            labels[day] = SPREAD
            continue
        rate = rates.rate_for(day.weekday())
        first_inside = min(inside_times)
        last_inside = max(inside_times)
        head_minutes = (first_inside - session.start).total_seconds() / 60.0
        tail_minutes = (session.end - last_inside).total_seconds() / 60.0
        head_expected = head_minutes * rate
        tail_expected = tail_minutes * rate
        expected_day_total = session.minutes() * rate
        tolerance = max(
            params.pin_tolerance_orders,
            params.pin_tolerance_fraction * expected_day_total,
        )
        start_pinned = head_expected <= tolerance
        end_pinned = tail_expected <= tolerance
        if start_pinned and end_pinned:
            labels[day] = FULL
        else:
            labels[day] = PARTIAL
    return labels
